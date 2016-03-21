/**
 * @file   criteo_parser.h
 * @brief  parse criteo ctr data format
 */
#pragma once
#include <limits>
#include <city.h>
#include "data/row_block.h"
#include "data/parser.h"
#include "data/strtonum.h"
#include "dmlc/recordio.h"
#include "sample.pb.h"
namespace dmlc {
namespace data {

/**
 * \brief criteo ctr dataset:
 * The columns are tab separeted with the following schema:
 *  <label> <integer feature 1> ... <integer feature 13>
 *  <categorical feature 1> ... <categorical feature 26>
 */



const size_t POLYNOM_CONSTANT = 27942141U;

template <class Callback, class IdClass>
void make_cartesian_product_of_namespaces(
        size_t index,
        const std::vector<size_t> & namespaces,
        const TrainingSample & event,
        double value,
        IdClass id,
        Callback resultCallback) {
    if (index >= namespaces.size()) {
        resultCallback(id, value);
        return;
    }
    const auto & cur_ns = event.namespaces(namespaces[index]);
    for (size_t featureIndex = 0, end = cur_ns.features_size();
                featureIndex < end;
                ++featureIndex) {
        make_cartesian_product_of_namespaces(
            index + 1,
            namespaces,
            event,
            value * cur_ns.values(featureIndex),
            id * POLYNOM_CONSTANT + cur_ns.features(featureIndex),
            resultCallback);
    }
}







template <typename IndexType>
class ProtoParser : public ParserImpl<IndexType> {
 public:
  explicit ProtoParser(InputSplit *source, const std::vector<std::vector<size_t>> & learn_namespaces)
      : bytes_read_(0), source_(source), learn_namespaces_(learn_namespaces) {
  }
  virtual ~ProtoParser() {
    delete source_;
  }

  virtual void BeforeFirst(void) {
    source_->BeforeFirst();
  }
  virtual size_t BytesRead(void) const {
    return bytes_read_;
  }
  virtual bool ParseNext(std::vector<RowBlockContainer<IndexType> > *data) {
    InputSplit::Blob chunk;
    if (!source_->NextChunk(&chunk)) return false;

    CHECK(chunk.size != 0);
    bytes_read_ += chunk.size;
    data->resize(1);
    RowBlockContainer<IndexType>& blk = (*data)[0];
    blk.Clear();
    RecordIOChunkReader reader(chunk);
    InputSplit::Blob record;
    while (reader.NextRecord(&record)) {

      TrainingSample sample;
      sample.ParseFromArray(record.dptr, record.size);
      // parse label
      blk.label.push_back(sample.label());


      for (auto & ns_set: learn_namespaces_) {

         make_cartesian_product_of_namespaces(0, ns_set, sample, 1., 0U,
                 [&blk](const IndexType & feature, float value) {
                    blk.index.push_back(feature);
                    blk.value.push_back(value);
                 }
         );
      }

      blk.offset.push_back(blk.index.size());
    }
    return true;
  }

 private:


  // number of bytes readed
  size_t bytes_read_;
  // source split that provides the data
  InputSplit *source_;
  std::vector<std::vector<size_t>> learn_namespaces_;
};

}  // namespace data
}  // namespace dmlc


