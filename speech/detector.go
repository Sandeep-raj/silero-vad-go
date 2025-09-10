package speech

// #cgo CFLAGS: -Wall -Werror -std=c99
// #cgo LDFLAGS: -lonnxruntime
// #include "ort_bridge.h"
import "C"

import (
	"fmt"
	"log/slog"
	"sync"
	"unsafe"
)

const (
	stateLen   = 2 * 1 * 128
	contextLen = 64
)

type LogLevel int

func (l LogLevel) OrtLoggingLevel() C.OrtLoggingLevel {
	switch l {
	case LevelVerbose:
		return C.ORT_LOGGING_LEVEL_VERBOSE
	case LogLevelInfo:
		return C.ORT_LOGGING_LEVEL_INFO
	case LogLevelWarn:
		return C.ORT_LOGGING_LEVEL_WARNING
	case LogLevelError:
		return C.ORT_LOGGING_LEVEL_ERROR
	case LogLevelFatal:
		return C.ORT_LOGGING_LEVEL_FATAL
	default:
		return C.ORT_LOGGING_LEVEL_WARNING
	}
}

const (
	LevelVerbose LogLevel = iota + 1
	LogLevelInfo
	LogLevelWarn
	LogLevelError
	LogLevelFatal
)

type DetectorConfig struct {
	// The path to the ONNX Silero VAD model file to load.
	ModelPath string
	// The sampling rate of the input audio samples. Supported values are 8000 and 16000.
	SampleRate int
	// The probability threshold above which we detect speech. A good default is 0.5.
	Threshold float32
	// The duration of silence to wait for each speech segment before separating it.
	MinSilenceDurationMs int
	// The padding to add to speech segments to avoid aggressive cutting.
	SpeechPadMs int
	// The loglevel for the onnx environment, by default it is set to LogLevelWarn.
	LogLevel LogLevel
}

func (c DetectorConfig) IsValid() error {
	if c.ModelPath == "" {
		return fmt.Errorf("invalid ModelPath: should not be empty")
	}

	if c.SampleRate != 8000 && c.SampleRate != 16000 {
		return fmt.Errorf("invalid SampleRate: valid values are 8000 and 16000")
	}

	if c.Threshold <= 0 || c.Threshold >= 1 {
		return fmt.Errorf("invalid Threshold: should be in range (0, 1)")
	}

	if c.MinSilenceDurationMs < 0 {
		return fmt.Errorf("invalid MinSilenceDurationMs: should be a positive number")
	}

	if c.SpeechPadMs < 0 {
		return fmt.Errorf("invalid SpeechPadMs: should be a positive number")
	}

	return nil
}

type Detector struct {
	api         *C.OrtApi
	env         *C.OrtEnv
	sessionOpts *C.OrtSessionOptions
	session     *C.OrtSession
	memoryInfo  *C.OrtMemoryInfo
	cStrings    map[string]*C.char

	cfg DetectorConfig

	state [stateLen]float32
	ctx   [contextLen]float32

	currSample int
	triggered  bool
	tempEnd    int
}

// global shared ONNX runtime environment
var (
	ortApi         *C.OrtApi
	ortEnv         *C.OrtEnv
	ortSessionOpts *C.OrtSessionOptions
	onnxInitOnce   sync.Once
	onnxInitErr    error
)

type sharedSession struct {
	session    *C.OrtSession
	memoryInfo *C.OrtMemoryInfo // optional shared memory info; you may still create per-detector meminfo
	refcount   int
}

// sessionManager maintains sessions per modelPath
var (
	smMu  sync.Mutex
	smMap = map[string]*sharedSession{}
)

// GetSharedSession returns (session, memoryInfo) and increments refcount.
// Caller MUST call ReleaseSharedSession when done.
func GetSharedSession(modelPath string) (*C.OrtSession, *C.OrtMemoryInfo, error) {
	if ortApi == nil || ortEnv == nil || ortSessionOpts == nil {
		return nil, nil, fmt.Errorf("onnx not initialized")
	}

	smMu.Lock()
	defer smMu.Unlock()

	ss, ok := smMap[modelPath]
	if ok {
		ss.refcount++
		return ss.session, ss.memoryInfo, nil
	}

	// create session
	cModelPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cModelPath))

	var status *C.OrtStatus
	var session *C.OrtSession
	status = C.OrtApiCreateSession(ortApi, ortEnv, cModelPath, ortSessionOpts, &session)
	defer C.OrtApiReleaseStatus(ortApi, status)
	if status != nil {
		return nil, nil, fmt.Errorf("failed to create session: %s", C.GoString(C.OrtApiGetErrorMessage(ortApi, status)))
	}

	// create memoryInfo (you may reuse one global meminfo instead)
	var mem *C.OrtMemoryInfo
	status = C.OrtApiCreateCpuMemoryInfo(ortApi, C.OrtArenaAllocator, C.OrtMemTypeDefault, &mem)
	defer C.OrtApiReleaseStatus(ortApi, status)
	if status != nil {
		// release session before returning
		C.OrtApiReleaseSession(ortApi, session)
		return nil, nil, fmt.Errorf("failed to create memory info: %s", C.GoString(C.OrtApiGetErrorMessage(ortApi, status)))
	}

	ss = &sharedSession{
		session:    session,
		memoryInfo: mem,
		refcount:   1,
	}
	smMap[modelPath] = ss

	return ss.session, ss.memoryInfo, nil
}

// ReleaseSharedSession decrements reference count and frees session when 0.
func ReleaseSharedSession(modelPath string) error {
	smMu.Lock()
	defer smMu.Unlock()

	ss, ok := smMap[modelPath]
	if !ok {
		return fmt.Errorf("no shared session for %s", modelPath)
	}

	ss.refcount--
	if ss.refcount > 0 {
		return nil
	}

	// free resources
	C.OrtApiReleaseMemoryInfo(ortApi, ss.memoryInfo)
	C.OrtApiReleaseSession(ortApi, ss.session)
	delete(smMap, modelPath)
	return nil
}

// InitOnnx initializes ONNX Runtime once per process and reuses env/session options.
func InitOnnx(logLevel LogLevel) error {
	onnxInitOnce.Do(func() {
		ortApi = C.OrtGetApi()
		if ortApi == nil {
			onnxInitErr = fmt.Errorf("failed to get OrtApi")
			return
		}

		// Create Env
		loggerName := C.CString("vad-global")
		defer C.free(unsafe.Pointer(loggerName))

		var status *C.OrtStatus
		status = C.OrtApiCreateEnv(ortApi, logLevel.OrtLoggingLevel(), loggerName, &ortEnv)
		defer C.OrtApiReleaseStatus(ortApi, status)
		if status != nil {
			onnxInitErr = fmt.Errorf("failed to create env: %s", C.GoString(C.OrtApiGetErrorMessage(ortApi, status)))
			return
		}

		// Create SessionOptions
		status = C.OrtApiCreateSessionOptions(ortApi, &ortSessionOpts)
		defer C.OrtApiReleaseStatus(ortApi, status)
		if status != nil {
			onnxInitErr = fmt.Errorf("failed to create session options: %s", C.GoString(C.OrtApiGetErrorMessage(ortApi, status)))
			return
		}

		// Configure threading and optimizations
		status = C.OrtApiSetIntraOpNumThreads(ortApi, ortSessionOpts, 1)
		defer C.OrtApiReleaseStatus(ortApi, status)
		if status != nil {
			onnxInitErr = fmt.Errorf("failed to set intra threads: %s", C.GoString(C.OrtApiGetErrorMessage(ortApi, status)))
			return
		}

		status = C.OrtApiSetInterOpNumThreads(ortApi, ortSessionOpts, 1)
		defer C.OrtApiReleaseStatus(ortApi, status)
		if status != nil {
			onnxInitErr = fmt.Errorf("failed to set inter threads: %s", C.GoString(C.OrtApiGetErrorMessage(ortApi, status)))
			return
		}

		status = C.OrtApiSetSessionGraphOptimizationLevel(ortApi, ortSessionOpts, C.ORT_ENABLE_ALL)
		defer C.OrtApiReleaseStatus(ortApi, status)
		if status != nil {
			onnxInitErr = fmt.Errorf("failed to set graph optimization: %s", C.GoString(C.OrtApiGetErrorMessage(ortApi, status)))
			return
		}
	})

	return onnxInitErr
}

func NewDetector(cfg DetectorConfig) (*Detector, error) {
	if err := cfg.IsValid(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	// Ensure ONNX is initialized
	if err := InitOnnx(cfg.LogLevel); err != nil {
		return nil, err
	}

	sd := Detector{
		cfg:         cfg,
		cStrings:    map[string]*C.char{},
		api:         ortApi,
		env:         ortEnv,
		sessionOpts: ortSessionOpts,
	}

	// Get shared session for the modelPath
	session, _, err := GetSharedSession(cfg.ModelPath)
	if err != nil {
		return nil, err
	}
	sd.session = session

	// sd.cStrings["modelPath"] = C.CString(cfg.ModelPath)
	// status := C.OrtApiCreateSession(ortApi, ortEnv, sd.cStrings["modelPath"], ortSessionOpts, &sd.session)
	// defer C.OrtApiReleaseStatus(ortApi, status)
	// if status != nil {
	// 	return nil, fmt.Errorf("failed to create session: %s", C.GoString(C.OrtApiGetErrorMessage(ortApi, status)))
	// }

	status = C.OrtApiCreateCpuMemoryInfo(ortApi, C.OrtArenaAllocator, C.OrtMemTypeDefault, &sd.memoryInfo)
	defer C.OrtApiReleaseStatus(ortApi, status)
	if status != nil {
		return nil, fmt.Errorf("failed to create memory info: %s", C.GoString(C.OrtApiGetErrorMessage(ortApi, status)))
	}

	// Input/Output names
	sd.cStrings["input"] = C.CString("input")
	sd.cStrings["sr"] = C.CString("sr")
	sd.cStrings["state"] = C.CString("state")
	sd.cStrings["stateN"] = C.CString("stateN")
	sd.cStrings["output"] = C.CString("output")

	return &sd, nil
}

// Segment contains timing information of a speech segment.
type Segment struct {
	// The relative timestamp in seconds of when a speech segment begins.
	SpeechStartAt float64
	// The relative timestamp in seconds of when a speech segment ends.
	SpeechEndAt float64
}

func (sd *Detector) Detect(pcm []float32) ([]Segment, error) {
	if sd == nil {
		return nil, fmt.Errorf("invalid nil detector")
	}

	windowSize := 512
	if sd.cfg.SampleRate == 8000 {
		windowSize = 256
	}

	if len(pcm) < windowSize {
		return nil, fmt.Errorf("not enough samples")
	}

	slog.Debug("starting speech detection", slog.Int("samplesLen", len(pcm)))

	minSilenceSamples := sd.cfg.MinSilenceDurationMs * sd.cfg.SampleRate / 1000
	speechPadSamples := sd.cfg.SpeechPadMs * sd.cfg.SampleRate / 1000

	var segments []Segment
	for i := 0; i < len(pcm)-windowSize; i += windowSize {
		speechProb, err := sd.infer(pcm[i : i+windowSize])
		if err != nil {
			return nil, fmt.Errorf("infer failed: %w", err)
		}

		sd.currSample += windowSize

		if speechProb >= sd.cfg.Threshold && sd.tempEnd != 0 {
			sd.tempEnd = 0
		}

		if speechProb >= sd.cfg.Threshold && !sd.triggered {
			sd.triggered = true
			speechStartAt := (float64(sd.currSample-windowSize-speechPadSamples) / float64(sd.cfg.SampleRate))

			// We clamp at zero since due to padding the starting position could be negative.
			if speechStartAt < 0 {
				speechStartAt = 0
			}

			slog.Debug("speech start", slog.Float64("startAt", speechStartAt))
			segments = append(segments, Segment{
				SpeechStartAt: speechStartAt,
			})
		}

		if speechProb < (sd.cfg.Threshold-0.15) && sd.triggered {
			if sd.tempEnd == 0 {
				sd.tempEnd = sd.currSample
			}

			// Not enough silence yet to split, we continue.
			if sd.currSample-sd.tempEnd < minSilenceSamples {
				continue
			}

			speechEndAt := (float64(sd.tempEnd+speechPadSamples) / float64(sd.cfg.SampleRate))
			sd.tempEnd = 0
			sd.triggered = false
			slog.Debug("speech end", slog.Float64("endAt", speechEndAt))

			if len(segments) < 1 {
				return nil, fmt.Errorf("unexpected speech end")
			}

			segments[len(segments)-1].SpeechEndAt = speechEndAt
		}
	}

	slog.Debug("speech detection done", slog.Int("segmentsLen", len(segments)))

	return segments, nil
}

func (sd *Detector) Reset() error {
	if sd == nil {
		return fmt.Errorf("invalid nil detector")
	}

	sd.currSample = 0
	sd.triggered = false
	sd.tempEnd = 0
	for i := 0; i < stateLen; i++ {
		sd.state[i] = 0
	}
	for i := 0; i < contextLen; i++ {
		sd.ctx[i] = 0
	}

	return nil
}

func (sd *Detector) SetThreshold(value float32) {
	sd.cfg.Threshold = value
}

func (sd *Detector) Destroy() error {
	if sd == nil {
		return fmt.Errorf("invalid nil detector")
	}

	C.OrtApiReleaseMemoryInfo(sd.api, sd.memoryInfo)
	if err := ReleaseSharedSession(sd.cfg.ModelPath); err != nil {
		return err
	}

	for _, ptr := range sd.cStrings {
		C.free(unsafe.Pointer(ptr))
	}
	return nil
}
