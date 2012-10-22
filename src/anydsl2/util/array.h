#ifndef ANYDSL2_ARRAY_H
#define ANYDSL2_ARRAY_H

#include <cstddef>
#include <cstring>
#include <iterator>
#include <vector>

#include <boost/functional/hash.hpp>

#include "anydsl2/util/assert.h"
#include "anydsl2/util/for_all.h"

namespace anydsl2 {

template<class T> class Array;

//------------------------------------------------------------------------------

template<class T>
class ArrayRef {
public:

    typedef const T* const_iterator;
    typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

    ArrayRef()
        : ptr_(0)
        , size_(0)
    {}
    ArrayRef(const ArrayRef<T>& ref)
        : ptr_(ref.ptr_)
        , size_(ref.size_)
    {}
    template<size_t N>
    ArrayRef(T (&array)[N])
        : ptr_(&array[0])
        , size_(N)
    {}
    ArrayRef(const std::vector<T>& vector)
        : ptr_(&*vector.begin())
        , size_(vector.size())
    {}
    ArrayRef(const T* ptr, size_t size)
        : ptr_(ptr)
        , size_(size)
    {}
    ArrayRef(const Array<T>& array)
        : ptr_(array.begin())
        , size_(array.size())
    {}

    const_iterator begin() const { return ptr_; }
    const_iterator end() const { return ptr_ + size_; }
    const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
    const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

    const T& operator [] (size_t i) const {
        assert(i < size() && "index out of bounds");
        return *(ptr_ + i);
    }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }

    T const& front() const { assert(!empty()); return ptr_[0]; }
    T const& back()  const { assert(!empty()); return ptr_[size_ - 1]; }

    ArrayRef<T> slice(size_t begin, size_t end) const { return ArrayRef<T>(ptr_ + begin, end - begin); }
    ArrayRef<T> slice_front(size_t end) const { return ArrayRef<T>(ptr_, end); }
    ArrayRef<T> slice_back(size_t begin) const { return ArrayRef<T>(ptr_ + begin, size_ - begin); }

    bool operator == (ArrayRef<T> other) const {
        if (size() != other.size())
            return false;

        bool result = true;
        for (size_t i = 0, e = size(); i != e && result; ++i)
            result &= ptr_[i] == other[i];

        return result;
    }

    template<class U>
    ArrayRef<U> cast() const { return ArrayRef<U>((const U*) ptr_, size_); }

private:

    const T* ptr_;
    size_t size_;
};

//------------------------------------------------------------------------------

template<class T>
class Array {
public:

    Array()
        : ptr_(0)
        , size_(0)
    {}
    explicit Array(size_t size)
        : ptr_(new T[size]())
        , size_(size)
    {}
    explicit Array(ArrayRef<T> ref)
        : ptr_(new T[ref.size()])
        , size_(ref.size())
    {
        std::memcpy(ptr_, ref.begin(), size() * sizeof(T));
    }
    Array(const Array<T>& array)
        : ptr_(new T[array.size()])
        , size_(array.size())
    {
        std::memcpy(ptr_, array.ptr_, size() * sizeof(T));
    }

    ~Array() { delete[] ptr_; }

    void alloc(size_t size) {
        assert(ptr_ == 0 && size_ == 0);
        ptr_ = new T[size]();
        size_ = size;
    };

    typedef T* iterator;
    typedef const T* const_iterator;
    typedef std::reverse_iterator<iterator> reverse_iterator;
    typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

    iterator begin() { return ptr_; }
    iterator end() { return ptr_ + size_; }
    reverse_iterator rbegin() { return const_reverse_iterator(end()); }
    reverse_iterator rend() { return const_reverse_iterator(begin()); }
    const_iterator begin() const { return ptr_; }
    const_iterator end() const { return ptr_ + size_; }
    const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
    const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

    T& operator [] (size_t i) { assert(i < size() && "index out of bounds"); return ptr_[i]; }
    T const& operator [] (size_t i) const { assert(i < size() && "index out of bounds"); return ptr_[i]; }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    bool valid() const { return ptr_; }

    T& front() const { assert(!empty()); return ptr_[0]; }
    T& back()  const { assert(!empty()); return ptr_[size_ - 1]; }

    ArrayRef<T> slice(size_t begin, size_t end) const { return ArrayRef<T>(ptr_ + begin, end - begin); }
    ArrayRef<T> slice_front(size_t end) const { return ArrayRef<T>(ptr_, end); }
    ArrayRef<T> slice_back(size_t begin) const { return ArrayRef<T>(ptr_ + begin, size_ - begin); }

    bool operator == (const Array<T>& other) const { return ArrayRef<T>(*this) == ArrayRef<T>(other); }

    void shrink(size_t newsize) { assert(newsize <= size_); size_ = newsize; }

    ArrayRef<T> ref() const { return ArrayRef<T>(ptr_, size_); }

private:

    Array<T>& operator = (const Array<T>& array);

    T* ptr_;
    size_t size_;
};

template<class T>
inline size_t hash_value(ArrayRef<T> aref) {
    size_t seed = 0;
    boost::hash_combine(seed, aref.size());

    for (size_t i = 0, e = aref.size(); i != e; ++i)
        boost::hash_combine(seed, aref[i]);

    return seed;
}

template<class T>
inline size_t hash_value(const Array<T>& array) { return hash_value(ArrayRef<T>(array)); }

//------------------------------------------------------------------------------

} // namespace anydsl2

#endif
