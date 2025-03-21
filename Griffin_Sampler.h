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
        float  volumeMult = 1.0f;        // Multiplies the note's velocity.
        float  panning = 0.0f;        // -1 (left) to +1 (right); used in a simple crossfade.
        float  startOffsetInSamples = 0.0f;     // Calculated sample start offset.
        float  endOffsetInSamples = 0.0f;     // Calculated sample end offset.
        bool   reverse = false;       // If true, playback goes in reverse.
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
            // Save the endpoint (for non-reverse, endpoint is higher than start)
            endOffset = endOff;

            // Compute playback rate (pitch shift) based on base delta and detune in cents.
            double centsFact = std::pow(2.0, (s.pitchOffsetCents / 1200.0));
            double effectiveSpeed = baseDelta * centsFact;
            phaseInc = (int64_t)std::llround(effectiveSpeed * FIXED_ONE);
            if (s.reverse)
            {
                phaseInc = -phaseInc;
                // For reverse playback, swap start and end: playback will end when phaseAcc <= endOff.
                endOffset = startOff;
            }
        }

        // Synthesis loop: perform sample read with linear interpolation between start and end.
        int vectorSynthesize(float* outL, float* outR, int blockSize)
        {
            if (!active)
                return 0;
            int processed = 0;
            const float leftGain = 0.5f * (1.f - settings.panning);
            const float rightGain = 0.5f * (1.f + settings.panning);
            while (processed < blockSize)
            {
                int64_t acc = phaseAcc >> FIXED_SHIFT;
                int idx = (int)acc;
                // Check if we've reached the designated end point.
                if (!settings.reverse)
                {
                    if (float(idx) >= endOffset)
                    {
                        active = false;
                        break;
                    }
                }
                else
                {
                    if (float(idx) <= endOffset)
                    {
                        active = false;
                        break;
                    }
                }
                float frac = float(phaseAcc & FIXED_MASK) / float(FIXED_ONE);
                float sampL = sourceL[idx] + frac * (sourceL[idx + 1] - sourceL[idx]);
                float sampR = sourceR[idx] + frac * (sourceR[idx + 1] - sourceR[idx]);
                outL[processed] += sampL * amplitude * leftGain;
                outR[processed] += sampR * amplitude * rightGain;
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

        std::vector<Voice*> activeVoices;
        PolyData<Voice, NV> voices;
        ExternalData sampleData;
        AudioBuffer<float> sampleBuffer;
        std::array<const float*, 2> sample{ nullptr, nullptr };

        std::array<float, 128> pitchRatios{};
        double sampleRate = 44100.0;
        double sampleRateRatio = 1.0;

        // Parameters for sample playback range (start and end positions as percentages)
        float sampleStartPercent = 0.0f;      // Relative start point in the sample [0, 1].
        float sampleEndPercent = 1.0f;        // Relative end point in the sample [0, 1].
        // Calculated absolute positions in samples.
        float sampleStartOffsetInSamples = 0.0f;
        float sampleEndOffsetInSamples = 0.0f;

        bool forceReverse = false;            // If true, playback is forced in reverse.
        double globalPitchOffsetFactor = 1.0;   // Acts as a pitch multiplier.
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
            activeVoices.clear();
            activeVoices.reserve(NV);
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
                settings.reverse = forceReverse;
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

            activeVoices.clear();
            for (auto& voice : voices)
            {
                if (voice.isActive)
                    activeVoices.push_back(&voice);
            }
            std::vector<float> tempBlockOutL(totalSamples, 0.f);
            std::vector<float> tempBlockOutR(totalSamples, 0.f);

            for (auto* vptr : activeVoices)
            {
                int n = vptr->playback.vectorSynthesize(tempBlockOutL.data(), tempBlockOutR.data(), totalSamples);
                if (!vptr->playback.active)
                    vptr->isActive = false;
            }
            for (int i = 0; i < totalSamples; ++i)
            {
                leftChannel[i] += tempBlockOutL[i];
                rightChannel[i] += tempBlockOutR[i];
            }
        }

        template <typename FrameDataType>
        void processFrame(FrameDataType&) {}

        // Parameter mapping:
        // Index 0: Pitch multiplier
        // Index 1: Sample Start (percent)
        // Index 2: Sample End (percent)
        // Index 3: Reverse flag
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
            else if constexpr (P == 3)
            {
                forceReverse = (v >= 0.5);
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
            {
                parameter::data reverseParam("Reverse", { 0.0, 1.0, 1.0 });
                registerCallback<3>(reverseParam);
                reverseParam.setDefaultValue(0.0);
                data.add(std::move(reverseParam));
            }
        }
    };

} // namespace project
