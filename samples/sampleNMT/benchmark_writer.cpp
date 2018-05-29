#include "benchmark_writer.h"

#include <iostream>

namespace nmtSample
{
    BenchmarkWriter::BenchmarkWriter()
        : mSampleCount(0)
        , mStartTS(std::chrono::high_resolution_clock::now())
    {
    }

    void BenchmarkWriter::write(
        int sampleCount,
        int maxOutputSequenceLength,
        const int * hOutputData,
        const int * hActualOutputSequenceLengths)
    {
        mSampleCount += sampleCount;
    }

    void BenchmarkWriter::initialize()
    {
        mStartTS = std::chrono::high_resolution_clock::now();
    }

    void BenchmarkWriter::finalize()
    {
        std::chrono::duration<float> sec = std::chrono::high_resolution_clock::now() - mStartTS;
        std::cout << mSampleCount << " sequences generated in " << sec.count() << " seconds, " << (mSampleCount / sec.count()) << " samples/sec" << std::endl;
    }

    std::string BenchmarkWriter::getInfo()
    {
        return "Benchmark Writer";
    }
}
