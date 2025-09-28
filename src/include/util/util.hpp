//
// Created by username2 on 2023/10/7.
//

#ifndef VECTORSETSEARCH_UTIL_HPP
#define VECTORSETSEARCH_UTIL_HPP
namespace VectorSetSearch::Method {

    bool FloatEqual(const float &f1, const float &f2) {
        return std::abs(f1 - f2) < 0.001f;
    }

    template<typename T>
    inline void PtrDelete(void *ptr) {
        T *tmp_ptr = static_cast<T *>(ptr);
        delete[] tmp_ptr;
    }

}
#endif //VECTORSETSEARCH_UTIL_HPP
