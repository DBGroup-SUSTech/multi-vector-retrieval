//
// Created by bianzheng on 2023/2/10.
//

#ifndef VECTORSETSEARCH_METHODBASE_HPP
#define VECTORSETSEARCH_METHODBASE_HPP

#include <vector>
#include <unordered_map>

#include "include/struct/TypeDef.hpp"

namespace VectorSetSearch {

    class BaseMethodIndex {
    public:

        // Delete copy constructor and assignment
        BaseMethodIndex(const BaseMethodIndex &) = delete;

        BaseMethodIndex &operator=(const BaseMethodIndex &) = delete;

        BaseMethodIndex() = default;

//        virtual bool addItem(const vector_set &item_embeddings, const std::uint32_t &externalID) = 0;

        virtual bool
        addItemBatch(const std::vector<vector_set> &item_embeddings_l, const std::vector<uint32_t> &externalID_l) = 0;

        virtual bool buildIndex() { return true; };

        virtual py::tuple searchKNN(const vector_set_list &qry_embeddings, const uint32_t &topk) = 0;

        virtual ~BaseMethodIndex() = default;

    };

}

#endif //VECTORSETSEARCH_METHODBASE_HPP
