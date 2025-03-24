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
        double pitchOffsetCents = 0.0;       // in cents; 0 means no detune
        float  volumeMult = 1.0f;            // multiplies velocity
        float  panning = 0.0f;               // -1 (L) to +1 (R)
        float  startOffsetInSamples = 0.0f;  // absolute start (inside the file)
        float  endOffsetInSamples = 0.0f;    // absolute end (inside the file)

        bool   loopMode = false;             // false = one-shot, true = loop
        float  xfadeLengthInSamples = 0.0f;  // actual crossfade length in samples (computed)
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
        float startOffset = 0.0f; // s
        float endOffset = 0.0f;   // e

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

            float sOff = s.startOffsetInSamples;
            float eOff = s.endOffsetInSamples;
            if (sOff < 0.f) sOff = 0.f;
            if (sOff > float(bufferLength - 1)) sOff = float(bufferLength - 1);
            if (eOff < 0.f) eOff = 0.f;
            if (eOff > float(bufferLength - 1)) eOff = float(bufferLength - 1);

            startOffset = sOff;
            endOffset = eOff;

            // Begin playback at sample start.
            phaseAcc = (int64_t)std::llround(sOff * FIXED_ONE);

            double centsFact = std::pow(2.0, (s.pitchOffsetCents / 1200.0));
            double effectiveSpeed = baseDelta * centsFact;
            phaseInc = (int64_t)std::llround(effectiveSpeed * FIXED_ONE);
        }

        // Synthesis loop: output one block of samples.
        // This implements standard crossfade looping:
        // - For positions p < (e - X): output sample normally.
        // - For p in [e - X, e): output crossfade between tail (at p)
        //   and head (at s + (p - (e - X))).
        // When p reaches or exceeds e, subtract (e - s - X) so that
        // the overlapping head region [s, s+X] is used.
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

            // Precomputed loop values:
            const bool loopEnabled = settings.loopMode;
            const float X = settings.xfadeLengthInSamples; // crossfade length
            const float sOff = startOffset;
            const float eOff = endOffset;
            const float L = (eOff > sOff) ? (eOff - sOff) : 0.f; // total region length

            while (processed < blockSize)
            {
                float p = float(phaseAcc >> FIXED_SHIFT) +
                    float(phaseAcc & FIXED_MASK) * invFixedOne;

                if (!loopEnabled && p >= eOff)
                {
                    active = false;
                    break;
                }

                float sampL = 0.f, sampR = 0.f;
                // If looping with crossfade and X > 0, check if we're in the crossfade zone.
                if (loopEnabled && X > 0.f && p >= (eOff - X))
                {
                    float alpha = (p - (eOff - X)) / X;
                    // Tail: sample at current position.
                    int idxTail = int(p);
                    float fracTail = p - float(idxTail);
                    float tailL = sourceL[idxTail] + fracTail * (sourceL[idxTail + 1] - sourceL[idxTail]);
                    float tailR = sourceR[idxTail] + fracTail * (sourceR[idxTail + 1] - sourceR[idxTail]);
                    // Head: corresponding sample from the beginning of the region.
                    float headPos = sOff + (p - (eOff - X));
                    int idxHead = int(headPos);
                    float fracHead = headPos - float(idxHead);
                    float headL = sourceL[idxHead] + fracHead * (sourceL[idxHead + 1] - sourceL[idxHead]);
                    float headR = sourceR[idxHead] + fracHead * (sourceR[idxHead + 1] - sourceR[idxHead]);
                    sampL = (1.f - alpha) * tailL + alpha * headL;
                    sampR = (1.f - alpha) * tailR + alpha * headR;
                }
                else
                {
                    int idx = int(p);
                    float frac = p - float(idx);
                    sampL = sourceL[idx] + frac * (sourceL[idx + 1] - sourceL[idx]);
                    sampR = sourceR[idx] + frac * (sourceR[idx + 1] - sourceR[idx]);
                }

                outL[processed] += sampL * ampLeft;
                outR[processed] += sampR * ampRight;
                processed++;

                phaseAcc += phaseInc;

                if (loopEnabled)
                {
                    float p_next = float(phaseAcc >> FIXED_SHIFT) +
                        float(phaseAcc & FIXED_MASK) * invFixedOne;
                    // When p_next exceeds the region, wrap it.
                    // The effective unique portion is (eOff - sOff - X).
                    if (p_next >= eOff)
                    {
                        float wrapAmt = (eOff - sOff - X);
                        if (wrapAmt < 0.f)
                            wrapAmt = 0.f;
                        p_next -= wrapAmt;
                        phaseAcc = (int64_t)std::llround(p_next * FIXED_ONE);
                    }
                }
            }
            return processed;
        }
    };

    struct Voice
    {
        int midiNote = 60;
        bool isActive = false;
        float velocity = 1.0f;
        SamplePlayback playback; // instance of sample playback

        // Reset the voice with a new SamplePlayback.
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

        // Playback range parameters (percent)
        float sampleStartPercent = 0.0f;
        float sampleEndPercent = 1.0f;
        float sampleStartOffsetInSamples = 0.0f;
        float sampleEndOffsetInSamples = 0.0f;

        double globalPitchOffsetFactor = 1.0;

        // Loop / crossfade parameters
        bool  loopMode = false;            // Parameter 3
        float xfadeFraction = 0.0f;         // Parameter 4 (0..1)
        float xfadeLengthInSamples = 0.0f;  // Computed crossfade length

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

            float regionLen = sampleEndOffsetInSamples - sampleStartOffsetInSamples;
            if (regionLen < 0.f)
                regionLen = 0.f;

            // Clamp crossfade: maximum allowed is half the region.
            float maxXfade = regionLen * 0.5f;
            float desiredXfade = xfadeFraction * regionLen;
            if (desiredXfade > maxXfade)
                desiredXfade = maxXfade;
            xfadeLengthInSamples = desiredXfade;
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

        // On note on, schedule a voice.
        void handleHiseEvent(HiseEvent& e)
        {
            if (e.isNoteOn())
            {
                auto& voice = voices.get();
                double baseDelta = pitchRatios[e.getNoteNumber()] * sampleRateRatio * globalPitchOffsetFactor;
                SampleSettings settings;
                settings.pitchOffsetCents = 0.0;
                settings.volumeMult = 1.0f;
                settings.panning = 0.0f;
                settings.startOffsetInSamples = sampleStartOffsetInSamples;
                settings.endOffsetInSamples = sampleEndOffsetInSamples;
                settings.loopMode = loopMode;
                settings.xfadeLengthInSamples = xfadeLengthInSamples;
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
        // 0: Pitch (semitones, -24 to 24)
        // 1: Sample Start (percent)
        // 2: Sample End (percent)
        // 3: Loop Mode (0..1; false if <0.5)
        // 4: Xfade Length (0..1)
        template <int P>
        void setParameter(double v)
        {
            if constexpr (P == 0)
            {
                globalPitchOffsetFactor = std::pow(2.0, v / 12.0);
                for (auto& voice : voices)
                {
                    if (voice.isActive)
                    {
                        double newBaseDelta = pitchRatios[voice.midiNote] * sampleRateRatio * globalPitchOffsetFactor;
                        double centsFact = std::pow(2.0, (voice.playback.settings.pitchOffsetCents / 1200.0));
                        double effectiveSpeed = newBaseDelta * centsFact;
                        voice.playback.phaseInc = (int64_t)std::llround(effectiveSpeed * FIXED_ONE);
                    }
                }
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
                loopMode = (v >= 0.5);
            }
            else if constexpr (P == 4)
            {
                xfadeFraction = (float)v;
                if (xfadeFraction < 0.f) xfadeFraction = 0.f;
                if (xfadeFraction > 1.f) xfadeFraction = 1.f;
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
                parameter::data pitchParam("Pitch (semitones)", { -24.0, 24.0, 0.01 });
                registerCallback<0>(pitchParam);
                pitchParam.setDefaultValue(0.0);
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
                parameter::data loopParam("Loop Mode", { 0.0, 1.0, 1.0 });
                registerCallback<3>(loopParam);
                loopParam.setDefaultValue(0.0);
                data.add(std::move(loopParam));
            }
            {
                parameter::data xfadeParam("Xfade Length", { 0.0, 1.0, 0.001 });
                registerCallback<4>(xfadeParam);
                xfadeParam.setDefaultValue(0.0);
                data.add(std::move(xfadeParam));
            }
        }
    };

} // namespace project
