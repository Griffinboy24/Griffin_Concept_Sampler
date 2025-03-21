#pragma once

#include <JuceHeader.h>
#include <array>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>
#include <new>

namespace project
{

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#if defined(_MSC_VER)
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE inline __attribute__((always_inline))
#endif

    using namespace juce;
    using namespace hise;
    using namespace scriptnode;

    static constexpr int FIXED_SHIFT = 16;
    static constexpr int64_t FIXED_ONE = (int64_t)1 << FIXED_SHIFT;
    static constexpr int64_t FIXED_MASK = FIXED_ONE - 1;

    // Settings for sample playback
    struct SampleSettings
    {
        double pitchOffsetCents = 0.0;       // In cents; 0 means no detune.
        float  volumeMult = 1.0f;            // Multiplies the note's velocity.
        float  panning = 0.0f;               // -1 (left) to +1 (right); used in a simple crossfade.
        float  startOffsetInSamples = 0.0f;  // Calculated sample start offset.
        float  endOffsetInSamples = 0.0f;    // Calculated sample end offset.
    };

    // Single instance of sample playback for a note.
    struct SamplePlayback
    {
        const float* sourceL = nullptr;
        const float* sourceR = nullptr;
        bool active = false;
        float amplitude = 1.0f;
        SampleSettings settings;
        int64_t phaseAcc = 0;
        int64_t phaseInc = 0;
        float endOffset = 0.0f;  // Absolute endpoint for playback

        SamplePlayback() noexcept = default;

        // Constructor for sample playback (note voice)
        SamplePlayback(const std::array<const float*, 2>& src, float amp,
            const SampleSettings& s, int bufferLength, double baseDelta)
        {
            sourceL = src[0];
            sourceR = src[1];
            settings = s;
            amplitude = amp * s.volumeMult;
            active = true;

            // Clamp start and end offsets to valid range.
            float startOff = s.startOffsetInSamples;
            float endOff = s.endOffsetInSamples;
            if (startOff < 0.f)
                startOff = 0.f;
            if (startOff > float(bufferLength - 1))
                startOff = float(bufferLength - 1);
            if (endOff < 0.f)
                endOff = 0.f;
            if (endOff > float(bufferLength - 1))
                endOff = float(bufferLength - 1);

            // Set phase accumulator at the start offset.
            phaseAcc = (int64_t)std::llround(startOff * FIXED_ONE);
            // Save the endpoint (for forward playback, endpoint is higher than start)
            endOffset = endOff;

            // Compute playback rate (pitch shift) based on base delta and detune in cents.
            double centsFact = std::pow(2.0, (s.pitchOffsetCents / 1200.0));
            double effectiveSpeed = baseDelta * centsFact;
            phaseInc = (int64_t)std::llround(effectiveSpeed * FIXED_ONE);
        }

        // Synthesis loop: perform sample read with linear interpolation.
        int vectorSynthesize(float* outL, float* outR, int blockSize)
        {
            if (!active)
                return 0;
            int processed = 0;
            const float invFixedOne = 1.f / FIXED_ONE;
            const float leftGain = 0.5f * (1.f - settings.panning);
            const float rightGain = 0.5f * (1.f + settings.panning);
            const float ampLeft = amplitude * leftGain;
            const float ampRight = amplitude * rightGain;

            while (processed < blockSize)
            {
                int idx = int(phaseAcc >> FIXED_SHIFT);
                // End playback if the sample index reaches or exceeds the endpoint.
                if (float(idx) >= endOffset)
                {
                    active = false;
                    break;
                }
                float frac = float(phaseAcc & FIXED_MASK) * invFixedOne;
                float sampL = sourceL[idx] + frac * (sourceL[idx + 1] - sourceL[idx]);
                float sampR = sourceR[idx] + frac * (sourceR[idx + 1] - sourceR[idx]);
                outL[processed] += sampL * ampLeft;
                outR[processed] += sampR * ampRight;
                phaseAcc += phaseInc;
                processed++;
            }
            return processed;
        }
    };

    struct Voice
    {
        int midiNote = 60;
        bool isActive = false;
        float velocity = 1.0f;
        SamplePlayback playback; // Single instance of sample playback for this note.

        // Reset the voice by instantiating a new SamplePlayback for the note.
        void reset(int note, float vel, const std::array<const float*, 2>& sample,
            int bufferLength, double baseDelta, const SampleSettings& settings)
        {
            midiNote = note;
            velocity = vel;
            isActive = true;
            new (&playback) SamplePlayback(sample, velocity, settings, bufferLength, baseDelta);
        }
    };

    template <int NV>
    struct Griffin_Sampler : public data::base
    {
        SNEX_NODE(Griffin_Sampler);
        struct MetadataClass { SN_NODE_ID("Griffin_Sampler"); };

        static constexpr bool isModNode() { return false; }
        static constexpr bool isPolyphonic() { return NV > 1; }
        static constexpr bool hasTail() { return true; }
        static constexpr bool isSuspendedOnSilence() { return false; }
        static constexpr int getFixChannelAmount() { return 2; }
        static constexpr int NumTables = 0;
        static constexpr int NumSliderPacks = 0;
        static constexpr int NumAudioFiles = 1;
        static constexpr int NumFilters = 0;
        static constexpr int NumDisplayBuffers = 0;

        PolyData<Voice, NV> voices;
        ExternalData sampleData;
        AudioBuffer<float> sampleBuffer;
        std::array<const float*, 2> sample{ nullptr, nullptr };

        std::array<float, 128> pitchRatios{};
        double sampleRate = 44100.0;
        double sampleRateRatio = 1.0;

        // Parameters for sample playback range (start and end positions as percentages)
        float sampleStartPercent = 0.0f;     // Relative start point in the sample [0, 1].
        float sampleEndPercent = 1.0f;       // Relative end point in the sample [0, 1].
        // Calculated absolute positions in samples.
        float sampleStartOffsetInSamples = 0.0f;
        float sampleEndOffsetInSamples = 0.0f;

        double globalPitchOffsetFactor = 1.0;  // Acts as a pitch multiplier.
        std::mt19937 randomGen;

        // Load external sample data.
        void setExternalData(const ExternalData& ed, int)
        {
            sampleData = ed;
            AudioSampleBuffer tempBuf = ed.toAudioSampleBuffer();
            int numSamples = tempBuf.getNumSamples();
            int numChannels = tempBuf.getNumChannels();
            if (numSamples <= 0)
            {
                int fallbackLen = 8;
                int chs = (numChannels > 0 ? numChannels : 2);
                AudioSampleBuffer fallback(chs, fallbackLen);
                fallback.clear();
                sampleBuffer.makeCopyOf(fallback, true);
            }
            else
            {
                sampleBuffer.makeCopyOf(tempBuf, true);
            }
            sample[0] = sampleBuffer.getReadPointer(0);
            if (numChannels > 1)
                sample[1] = sampleBuffer.getReadPointer(1);
            else
                sample[1] = sample[0];
            updateDerivedParameters();
        }

        void updateDerivedParameters()
        {
            int currentSampleLength = sampleBuffer.getNumSamples();
            if (currentSampleLength < 1)
                currentSampleLength = 1;
            sampleStartOffsetInSamples = sampleStartPercent * float(currentSampleLength - 1);
            sampleEndOffsetInSamples = sampleEndPercent * float(currentSampleLength - 1);
        }

        void reset()
        {
            for (auto& voice : voices)
                voice.isActive = false;
        }

        void prepare(PrepareSpecs specs)
        {
            sampleRate = specs.sampleRate;
            initPitchRatios();
            updateDerivedParameters();
            voices.prepare(specs);
            std::random_device rd;
            randomGen.seed(rd());
        }

        // On note on, schedule a single voice.
        void handleHiseEvent(HiseEvent& e)
        {
            if (e.isNoteOn())
            {
                auto& voice = voices.get();
                double baseDelta = pitchRatios[e.getNoteNumber()] * sampleRateRatio * globalPitchOffsetFactor;
                // Prepare sample playback settings.
                SampleSettings settings;
                settings.pitchOffsetCents = 0.0; // No detune by default.
                settings.volumeMult = 1.0f;
                settings.panning = 0.0f;
                settings.startOffsetInSamples = sampleStartOffsetInSamples;
                settings.endOffsetInSamples = sampleEndOffsetInSamples;
                voice.reset(e.getNoteNumber(), e.getFloatVelocity(), sample,
                    sampleBuffer.getNumSamples(), baseDelta, settings);
            }
        }

        template <typename ProcessDataType>
        void process(ProcessDataType& data)
        {
            auto& fixData = data.template as<ProcessData<getFixChannelAmount()>>();
            auto audioBlock = fixData.toAudioBlock();
            auto* leftChannel = audioBlock.getChannelPointer(0);
            auto* rightChannel = audioBlock.getChannelPointer(1);
            int totalSamples = data.getNumSamples();
            if (sampleBuffer.getNumSamples() == 0)
            {
                audioBlock.clear();
                return;
            }
            std::fill(leftChannel, leftChannel + totalSamples, 0.f);
            std::fill(rightChannel, rightChannel + totalSamples, 0.f);

            for (auto& voice : voices)
            {
                if (!voice.isActive)
                    continue;
                int n = voice.playback.vectorSynthesize(leftChannel, rightChannel, totalSamples);
                if (!voice.playback.active)
                    voice.isActive = false;
            }
        }

        template <typename FrameDataType>
        void processFrame(FrameDataType&) {}

        // Parameter mapping:
        // Index 0: Pitch multiplier
        // Index 1: Sample Start (percent)
        // Index 2: Sample End (percent)
        template <int P>
        void setParameter(double v)
        {
            if constexpr (P == 0)
            {
                globalPitchOffsetFactor = v;
            }
            else if constexpr (P == 1)
            {
                sampleStartPercent = (float)v;
                updateDerivedParameters();
            }
            else if constexpr (P == 2)
            {
                sampleEndPercent = (float)v;
                updateDerivedParameters();
            }
        }

        void initPitchRatios()
        {
            for (int i = 0; i < 128; ++i)
                pitchRatios[i] = std::pow(2.0f, float(i - 60) / 12.0f);
        }

        void createParameters(ParameterDataList& data)
        {
            {
                parameter::data pitchParam("Pitch (multiplier)", { 0.25, 4.0, 0.01 });
                registerCallback<0>(pitchParam);
                pitchParam.setDefaultValue(1.0);
                data.add(std::move(pitchParam));
            }
            {
                parameter::data startParam("Sample Start", { 0.0, 1.0, 0.001 });
                registerCallback<1>(startParam);
                startParam.setDefaultValue(0.0);
                data.add(std::move(startParam));
            }
            {
                parameter::data endParam("Sample End", { 0.0, 1.0, 0.001 });
                registerCallback<2>(endParam);
                endParam.setDefaultValue(1.0);
                data.add(std::move(endParam));
            }
        }
    };

} // namespace project
