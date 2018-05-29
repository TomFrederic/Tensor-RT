#ifndef SAMPLE_NMT_BENCHMARK_WRITER_
#define SAMPLE_NMT_BENCHMARK_WRITER_

#include <memory>
#include <chrono>

#include "data_writer.h"

namespace nmtSample
{
    /** \class BenchmarkWriter
    *
    * \brief all it does is to measure the performance of sequence generation
    *
    */
    class BenchmarkWriter : public DataWriter
    {
    public:
        BenchmarkWriter();

        virtual void write(
            int sampleCount,
            int maxOutputSequenceLength,
            const int * hOutputData,
            const int * hActualOutputSequenceLengths);

        virtual void initialize();

        virtual void finalize();

        virtual std::string getInfo();

        virtual ~BenchmarkWriter() = default;

    private:
        int mSampleCount;
        std::chrono::high_resolution_clock::time_point mStartTS;
    };
}

#endif // SAMPLE_NMT_BENCHMARK_WRITER_
