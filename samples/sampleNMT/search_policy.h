#ifndef SAMPLE_NMT_SEARCH_POLICY_
#define SAMPLE_NMT_SEARCH_POLICY_

#include <memory>

#include "likelihood_combination_operator.h"
#include "component.h"

namespace nmtSample
{
    /** \class SearchPolicy
    *
    * \brief processes the results of the one iteration of the generator and produces input for the next iteration
    *
    */
    class SearchPolicy : public Component
    {
    public:
        typedef std::shared_ptr<SearchPolicy> ptr;

        SearchPolicy(int endSequenceId, LikelihoodCombinationOperator::ptr likelihoodCombinationOperator);

        virtual void initialize(
            int sampleCount,
            int * maxOutputSequenceLengths) = 0;

        virtual void processTimestep(
            int sampleCount,
            const float * hLikelihoods,
            const int * hVocabularyIndices,
            int * hSourceRayIndices,
            int * hSourceRayOptionIndices) = 0;

        virtual bool haveMoreWork() = 0;

        virtual void readGeneratedResult(
            int sampleCount,
            int maxOutputSequenceLength,
            int * hOutputData,
            int * hActualOutputSequenceLengths) = 0;

        virtual ~SearchPolicy() = default;

    protected:
        int mEndSequenceId;
        LikelihoodCombinationOperator::ptr mLikelihoodCombinationOperator;
    };
}

#endif // SAMPLE_NMT_SEARCH_POLICY_
