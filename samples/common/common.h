#ifndef _TRT_COMMON_H_
#define _TRT_COMMON_H_
#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

using namespace std;

#define CHECK(status)                             \
    do                                            \
    {                                             \
        auto ret = (status);                      \
        if (ret != 0)                             \
        {                                         \
            std::cout << "Cuda failure: " << ret; \
            abort();                              \
        }                                         \
    } while (0)

// Logger for GIE info/warning/errors
class Logger : public nvinfer1::ILogger
{
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) override
    {
        // suppress info-level messages
        if (severity == Severity::kINFO) return;

        switch (severity)
        {
        case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
        case Severity::kERROR: std::cerr << "ERROR: "; break;
        case Severity::kWARNING: std::cerr << "WARNING: "; break;
        case Severity::kINFO: std::cerr << "INFO: "; break;
        default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }
};

inline std::string locateFile(const std::string& input, const std::vector<std::string>& directories)
{
    std::string file;
    const int MAX_DEPTH{10};
    bool found{false};
    for (auto& dir : directories)
    {
        file = dir + input;
        for (int i = 0; i < MAX_DEPTH && !found; i++)
        {
            std::ifstream checkFile(file);
            found = checkFile.is_open();
            if (found) break;
            file = "../" + file;
        }
        if (found) break;
        file.clear();
    }

    assert(!file.empty() && "Could not find a file due to it not existing in the data directory.");
    return file;
}

inline void readPGMFile(const std::string& fileName, uint8_t* buffer, int inH, int inW)
{
    std::ifstream infile(fileName, std::ifstream::binary);
    assert(infile.is_open() && "Attempting to read from a file that is not open.");
    std::string magic, h, w, max;
    infile >> magic >> h >> w >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(buffer), inH * inW);
}

namespace samples_common
{

inline bool isDebug()
{
    return (std::getenv("TENSORRT_DEBUG") ? true : false);
}

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj) {
            obj->destroy();
        }
    }
};

template <typename T>
inline std::shared_ptr<T> infer_object(T* obj)
{
    if (!obj) {
        throw std::runtime_error("Failed to create object");
    }
    return std::shared_ptr<T>(obj, InferDeleter());
}

template <class Iter>
inline std::vector<size_t> argsort(Iter begin, Iter end, bool reverse = false)
{
    std::vector<size_t> inds(end - begin);
    std::iota(inds.begin(), inds.end(), 0);
    if (reverse) {
        std::sort(inds.begin(), inds.end(), [&begin](size_t i1, size_t i2) {
            return begin[i2] < begin[i1];
        });
    }
    else
    {
        std::sort(inds.begin(), inds.end(), [&begin](size_t i1, size_t i2) {
            return begin[i1] < begin[i2];
        });
    }
    return inds;
}

inline bool readReferenceFile(const std::string& fileName, std::vector<std::string>& refVector)
{
    std::ifstream infile(fileName);
    if (!infile.is_open()) {
        cout << "ERROR: readReferenceFile: Attempting to read from a file that is not open." << endl;
        return false;
    }
    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        refVector.push_back(line);
    }
    infile.close();
    return true;
}

template <typename result_vector_t>
inline std::vector<std::string> classify(const vector<string>& refVector, const result_vector_t& output, const size_t topK)
{
    auto inds = samples_common::argsort(output.cbegin(), output.cend(), true);
    std::vector<std::string> result;
    for (size_t k = 0; k < topK; ++k) {
        result.push_back(refVector[inds[k]]);
    }
    return result;
}

//...LG returns top K indices, not values.
template <typename T>
inline vector<size_t> topK(const vector<T> inp, const size_t k)
{
    vector<size_t> result;
    std::vector<size_t> inds = samples_common::argsort(inp.cbegin(), inp.cend(), true);
    result.assign(inds.begin(), inds.begin()+k);
    return result;
}

template <typename T>
inline bool readASCIIFile(const string& fileName, const size_t size, vector<T>& out)
{
    std::ifstream infile(fileName);
    if (!infile.is_open()) {
        cout << "ERROR readASCIIFile: Attempting to read from a file that is not open." << endl;
        return false;
    }
    out.clear();
    out.reserve(size);
    out.assign(std::istream_iterator<T>(infile), std::istream_iterator<T>());
    infile.close();
    return true;
}

template <typename T>
inline bool writeASCIIFile(const string& fileName, const vector<T>& in)
{
    std::ofstream outfile(fileName);
    if (!outfile.is_open()) {
        cout << "ERROR: writeASCIIFile: Attempting to write to a file that is not open." << endl;
        return false;
    }
    for (auto fn : in) {
        outfile << fn << " ";
    }
    outfile.close();
    return true;
}

inline void print_version()
{
//... This can be only done after statically linking this support into parserONNX.library
#if 0
    std::cout << "Parser built against:" << std::endl;
    std::cout << "  ONNX IR version:  " << nvonnxparser::onnx_ir_version_string(onnx::IR_VERSION) << std::endl;
#endif
    std::cout << "  TensorRT version: "
              << NV_TENSORRT_MAJOR << "."
              << NV_TENSORRT_MINOR << "."
              << NV_TENSORRT_PATCH << "."
              << NV_TENSORRT_BUILD << std::endl;
}

inline string getFileType(const string& filepath)
{
    return filepath.substr(filepath.find_last_of(".") + 1);
}

inline string toLower(const string& inp)
{
    string out = inp;
    std::transform(out.begin(), out.end(), out.begin(), ::tolower);
    return out;
}

} // namespace samples_common

#endif // _TRT_COMMON_H_
