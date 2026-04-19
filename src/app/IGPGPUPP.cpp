//
// Created by bianzheng on 2025/12/26.
//
#include <cstdint>
typedef uint32_t __u32;
typedef uint64_t __u64;
#include <string>
#include <parser.hpp>
#include <cnpy.h>
#include <spdlog/spdlog.h>

#include "impl/IGPGPUPP.hpp"
#include "include/util/CPPIOUtil.hpp"

void configure(cmd_line_parser::parser& parser) {
  parser.add("username", "Username", "-username", true);
  parser.add("dataset", "Dataset", "-dataset", true);
  parser.add("method_name", "Method name", "-method-name", true);

  parser.add("n_centroid", "Number of centroid of the index",
             "-num-centroid", true);
  parser.add("n_bit", "Number of bits in the SQ index",
             "-num-bit", true);

  parser.add("topk", "Number of nearest neighbours.", "-topk", true);
  parser.add("nprobe", "number of probe", "-nprobe", true);
  parser.add("probe_topk", "topk of the probe", "-probe_topk", true);

  parser.add("is_profile", "is profile the result", "-is_profile", true);
}

std::map<std::string, std::string> getFilenameMap(
    const std::string& username, const std::string& dataset,
    const std::string& method_name, const std::string& build_index_suffix) {

  char index_path[512];
  sprintf(index_path,
          "/home/%s/Dataset/multi-vector-retrieval/Index/%s/%s-%s",
          username.c_str(), dataset.c_str(),
          method_name.c_str(), build_index_suffix.c_str());

  char embedding_path[512];
  sprintf(embedding_path,
          "/home/%s/Dataset/multi-vector-retrieval/Embedding/%s",
          username.c_str(), dataset.c_str());

  char answer_path[512];
  sprintf(answer_path,
          "/home/%s/Dataset/multi-vector-retrieval/Result/answer",
          username.c_str());

  std::map<std::string, std::string> filename_m;

  std::string centroid_l_fname = std::string(index_path) + "/centroid_l.npy";
  std::string vq_code_l_fname = std::string(index_path) + "/vq_code_l.npy";
  std::string weight_l_fname = std::string(index_path) + "/weight_l.npy";
  std::string residual_code_l_fname =
      std::string(index_path) + "/residual_code_l.npy";

  std::string ivf_index_fname = std::string(index_path) + "/ivf.index";

  std::string query_l_fname = std::string(embedding_path) +
                              "/query_embedding.npy";
  std::string item_n_vec_l_fname =
      std::string(index_path) + "/item_n_vec_l.npy";

  return {{"centroid_l", centroid_l_fname}, {"vq_code_l", vq_code_l_fname},
          {"weight_l", weight_l_fname},
          {"residual_code_l", residual_code_l_fname},
          {"ivf_index", ivf_index_fname},
          {"query_l", query_l_fname}, {"item_n_vec_l", item_n_vec_l_fname},
          {"answer_path", std::string(answer_path)}};
}

/*
 * read the config, read the different data config in the file,
 * test in a batch, then read the config
 *
 * input is the index and query, output is two file: perforamnce.tsv in answer and the search_result.tsv in answer
 */
int main(int argc, char** argv) {

  cmd_line_parser::parser parser(argc, argv);
  configure(parser);
  bool success = parser.parse();
  if (!success) {
    spdlog::error("Error parsing command line");
    return 1;
  }

  const std::string username = parser.get<std::string>("username");
  const std::string dataset = parser.get<std::string>("dataset");
  const std::string method_name = parser.get<std::string>("method_name");

  const uint32_t n_centroid = parser.get<uint32_t>("n_centroid");
  const uint32_t n_bit = parser.get<uint32_t>("n_bit");
  char build_index_suffix_buffer[256];
  std::sprintf(build_index_suffix_buffer, "n_centroid_%u-n_bit_%u", n_centroid,
               n_bit);
  const std::string build_index_suffix = build_index_suffix_buffer;

  const uint32_t topk = parser.get<uint32_t>("topk");
  const uint32_t nprobe = parser.get<uint32_t>("nprobe");
  const uint32_t probe_topk = parser.get<uint32_t>("probe_topk");

  const bool is_profile = parser.get<bool>("is_profile");

  // read the index
  std::map<std::string, std::string> filename_m = getFilenameMap(
      username, dataset, method_name, build_index_suffix);

  const std::string item_n_vec_l_fname = filename_m["item_n_vec_l"];
  VectorSetSearch::Method::IGPGPUPP index(item_n_vec_l_fname);

  const std::string centroid_l_fname = filename_m["centroid_l"];
  const std::string vq_code_l_fname = filename_m["vq_code_l"];
  const std::string weight_l_fname = filename_m["weight_l"];
  const std::string residual_code_l_fname = filename_m["residual_code_l"];
  const std::string ivf_index_fname = filename_m["ivf_index"];

  index.buildIndex(centroid_l_fname, vq_code_l_fname, ivf_index_fname,
                   weight_l_fname,
                   residual_code_l_fname,
                   n_centroid, n_bit);

  const std::string query_l_fname = filename_m["query_l"];

  cnpy::NpyArray query_l_npy = cnpy::npy_load(query_l_fname);
  assert(query_l_npy.word_size == sizeof(float));
  assert(query_l_npy.shape.size() == 3);
  const uint32_t n_query = query_l_npy.shape[0];
  const uint32_t query_n_vec = query_l_npy.shape[1];
  assert(index._vec_dim == query_l_npy.shape[2]);
  const float* query_l = query_l_npy.data<float>();

  spdlog::info("start search");
  const VectorSetSearch::Method::IGPGPUPPResult result = index.search(
      query_l, n_query, query_n_vec, topk,
      nprobe, probe_topk);
  spdlog::info("end search");

  const std::string answer_path = filename_m["answer_path"];
  char retrieval_suffix_str[256];
  sprintf(retrieval_suffix_str, "nprobe_%d-probe_topk_%d",
          nprobe, probe_topk);

  const std::vector<uint32_t> actual_queryID_l = VectorSetSearch::readQueryID(
      username, dataset);
  assert(actual_queryID_l.size() == result.n_query);

  char retrieval_answer_filename[512];
  if (is_profile) {
    sprintf(retrieval_answer_filename, "%s/%s-%s-top%d-%s-%s-profile.tsv",
            answer_path.c_str(), dataset.c_str(), method_name.c_str(),
            topk, build_index_suffix.c_str(), retrieval_suffix_str);
  } else {
    sprintf(retrieval_answer_filename, "%s/%s-%s-top%d-%s-%s.tsv",
            answer_path.c_str(), dataset.c_str(), method_name.c_str(),
            topk, build_index_suffix.c_str(), retrieval_suffix_str);
  }

  std::ofstream answer_out;
  answer_out.open(retrieval_answer_filename);

  for (uint32_t qID = 0; qID < n_query; qID++) {
    for (uint32_t rank = 0; rank < topk; rank++) {
      answer_out << actual_queryID_l[qID] << "\t"
                 << result.result_ID_l[qID * topk + rank] << "\t"
                 << rank + 1 << "\t"
                 << result.result_score_l[qID * topk + rank] << std::endl;
    }
  }

  char performance_filename[512];
  if (is_profile) {
    sprintf(performance_filename, "%s/%s-%s-performance-top%d-%s-%s-profile.tsv",
            answer_path.c_str(), dataset.c_str(), method_name.c_str(),
            topk, build_index_suffix.c_str(), retrieval_suffix_str);
  }else {
    sprintf(performance_filename, "%s/%s-%s-performance-top%d-%s-%s.tsv",
            answer_path.c_str(), dataset.c_str(), method_name.c_str(),
            topk, build_index_suffix.c_str(), retrieval_suffix_str);
  }

  std::ofstream performance_out;
  performance_out.open(performance_filename);

  performance_out << "search_time" << std::endl;
  for (uint32_t qID = 0; qID < n_query; qID++) {
    performance_out << result.compute_time_l[qID] << std::endl;
  }
  performance_out.close();

  char performance_single_result_filename[512];
  if (is_profile) {
    sprintf(performance_single_result_filename, "%s/%s-%s-performance-top%d-%s-%s-profile.txt",
            answer_path.c_str(), dataset.c_str(), method_name.c_str(),
            topk, build_index_suffix.c_str(), retrieval_suffix_str);
  }else {
    sprintf(performance_single_result_filename, "%s/%s-%s-performance-top%d-%s-%s.txt",
            answer_path.c_str(), dataset.c_str(), method_name.c_str(),
            topk, build_index_suffix.c_str(), retrieval_suffix_str);
  }
  std::ofstream performance_single_result_out;
  performance_single_result_out.open(performance_single_result_filename);
  performance_single_result_out <<
                                "batch_query_time(s)\t" << result.batch_query_time << std::endl;
  performance_single_result_out <<
                                "n_query\t" << result.n_query << std::endl;
  performance_single_result_out <<
                                "topk\t" << result.topk << std::endl;
  performance_single_result_out.close();

  return 0;
}