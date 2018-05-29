#ifndef SAMPLE_NMT_BEAM_SEARCH_POLICY_
#define SAMPLE_NMT_BEAM_SEARCH_POLICY_

#include "search_policy.h"

#include <vector>

namespace nmtSample
{
    /** \class BeamSearchPolicy
    *
    * \brief processes the results of one iteration of the generator with beam search and produces input for the next iteration
    *
    */
    class BeamSearchPolicy : public SearchPolicy
    {
    public:
        BeamSearchPolicy(
            int endSequenceId,
            LikelihoodCombinationOperator::ptr likelihoodCombinationOperator,
            int beamWidth);

        virtual void initialize(
            int sampleCount,
            int * maxOutputSequenceLengths);

        virtual void processTimestep(
            int sampleCount,
            const float * hLikelihoods,
            const int * hVocabularyIndices,
            int * hSourceRayIndices,
            int * hSourceRayOptionIndices);

        virtual bool haveMoreWork();

        virtual void readGeneratedResult(
            int sampleCount,
            int maxOutputSequenceLength,
            int * hOutputData,
            int * hActualOutputSequenceLengths);

        virtual std::string getInfo();

        struct Ray
        {
            int vocabularyId;
            int backtrackId;
        };

        struct Option
        {
            int vocabularyId;
            int originalRayId;
            int optionInRayId;
            float likelihood;
        };

    protected:
        void backtrack(
            int lastTimestepId,
            int sampleId,
            int lastTimestepRayId,
            int * hOutputData,
            int lastTimestepWriteId) const;

    protected:
        int mBeamWidth;
        std::vector<bool> mValidRays;
        std::vector<float> mCurrentLikelihoods;
        std::vector<Ray> mBeamSearchTable;
        int mSampleCount;
        std::vector<int> mMaxOutputSequenceLengths;
        int mTimestepId;

        std::vector<std::vector<int>> mCandidates;
        std::vector<float> mCandidateLikelihoods;
    };
}

#endif // SAMPLE_NMT_BEAM_SEARCH_POLICY_
