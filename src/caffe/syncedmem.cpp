#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

#include <boost/thread.hpp> 

#include <stdio.h>
#include <stdlib.h>
#include <execinfo.h>
#include <cxxabi.h>
/** Print a demangled stack backtrace of the caller function to FILE* out. */
inline std::string print_stacktrace(FILE *out = stderr, unsigned int max_frames = 63) {
  fprintf(out, "stack trace begin:\n");

  void* addrlist[max_frames + 1];// storage array for stack trace address data
  int addrlen = backtrace(addrlist, sizeof(addrlist) / sizeof(void*));  // retrieve current stack addresses

  if (addrlen == 0) {
    fprintf(out, "  <empty, possibly corrupt>\n"); 
  }
  else {

    // resolve addresses into strings containing "filename(function+address)",
    // this array must be free()-ed
    char** symbollist = backtrace_symbols(addrlist, addrlen);

    size_t funcnamesize = 256;// allocate string which will be filled with the demangled function name
    char* funcname = (char*)malloc(funcnamesize);

    // iterate over the returned symbol lines. skip the first, it is the
    // address of this function.
    for (int i = 1; i < addrlen; i++) {
      char *begin_name = 0, *begin_offset = 0, *end_offset = 0;

      // find parentheses and +address offset surrounding the mangled name:
      // ./module(function+0x15c) [0x8048a6d]
      for (char *p = symbollist[i]; *p; ++p) {
        if (*p == '(')
          begin_name = p;
        else if (*p == '+')
          begin_offset = p;
        else if (*p == ')' && begin_offset) {
          end_offset = p;
          break;
        }
      }

      if (begin_name && begin_offset && end_offset      && begin_name < begin_offset) {
        *begin_name++ = '\0';
        *begin_offset++ = '\0';
        *end_offset = '\0';

        // mangled name is now in [begin_name, begin_offset) and caller
        // offset in [begin_offset, end_offset). now apply
        // __cxa_demangle():

        int status;
        char* ret = abi::__cxa_demangle(begin_name,
          funcname, &funcnamesize, &status);
        if (status == 0) {
          funcname = ret; // use possibly realloc()-ed string
          fprintf(out, "  %s : %s+%s\n",
            symbollist[i], funcname, begin_offset);
        }
        else {
          // demangling failed. Output function name as a C function with
          // no arguments.
          fprintf(out, "  %s : %s()+%s\n",
            symbollist[i], begin_name, begin_offset);
        }
      }
      else {

        fprintf(out, "  %s\n", symbollist[i]);// couldn't parse the line? print the whole line.
      }
    }

    free(funcname);
    free(symbollist);
  }

  return std::string("stack trace end");
}



DEFINE_int32(lms, -1,
  "Optional; large-model support (LMS) minimum size in KB."
  "any memory block larger than this will be subjet to LMS optimization"
  "by default (-1), disabled");

DEFINE_double(lms_frac, 0.0,
  "Optional: large-model support (LMS) enabling fraction of total memory"
  "by default (0.0), disabled");


namespace caffe {

  ///////////////////////////////////////////////////////////////////////////////////
  //LMS memory cache
  boost::mutex      _cache_mtx;

  class mem_mgr {
  public:
    size_t  _allocated;
    size_t  _available;
    std::vector<size_t>  _demanded;
    size_t  _big_alloc;
    size_t  _cpu2gpu;
    size_t  _gpu2cpu;
    mem_mgr() :
      _allocated(0), _available(0), _big_alloc(16), _cpu2gpu(0), _gpu2cpu(0) {

    }

    ~mem_mgr() {
      bool lms_on = FLAGS_lms >= 0;
      if (lms_on)
        dump();
    }

    void dump() {
      //std::cout << "[LMS] ==============================mem_mgr==============================" << std::endl;
      //std::cout << "[LMS] "
      //  << " _demanded=" << _demanded
      //  << " _allocated=" << _allocated
      //  << " (" << double(_allocated) / _demanded << ")"
      //  << std::endl << "[LMS] "
      //  << " _cpu2gpu=" << _cpu2gpu
      //  << " _gpu2cpu=" << _gpu2cpu
      //  << std::flush << std::endl;
    }

    class mem_info {
    public:
      size_t  _id;
      size_t  _s;
    };

    std::string dump(size_t id) {
      LOG(INFO) << "[LMS] mem_mgr[" << this << "] " << __FUNCTION__ << " ======= begin ========"
        << std::flush;

      std::vector<std::pair<char*, size_t> >& id_vector = _cache_map[id];

      if (!id_vector.empty()) {
        for (int i = 0; i < id_vector.size(); ++i) {
          if (id_vector[i].second < 1024 || i != id_vector.size() - 1) continue;

          LOG(INFO) << "[LMS] "
            << "dev=" << id
            << " mem: i=" << i
            << " ptr=" << size_t(id_vector[i].first)
            << " size=" << id_vector[i].second;
        }
      }

      return std::string(" ======= end ========");
    }

    bool get_mem(char** ret, size_t id, size_t s) {

      char* raw_mem; 
      bool alloced = false;

      if (!search(&raw_mem, id, s + sizeof(mem_info))) {
        size_t max_s = 0;
        std::map<size_t, size_t>& id_dist_map = _dist_map[id];
        for (std::map<size_t, size_t>::iterator itr = id_dist_map.begin(); itr != id_dist_map.end(); ++itr) {

          size_t maybe_size = itr->first* itr->second;

          max_s = std::max(max_s, itr->first);
          DLOG(INFO) << "[LMS] size=" << itr->first << " count=" << itr->second << " maybe_size=" << maybe_size;
        }

        size_t alloc_size = s + sizeof(mem_info)*_big_alloc;

        if (!id_dist_map.empty()) {
          std::map<size_t, size_t>::iterator itr = id_dist_map.find(s);
          ++itr;
          if (itr != id_dist_map.end() && (itr->first - s) / double(s) <= 0.7)
            alloc_size = itr->first + sizeof(mem_info)*_big_alloc;
        }

        allocate(&raw_mem, alloc_size);
        deallocate(raw_mem, id, alloc_size);
        search(&raw_mem, id, s + sizeof(mem_info));

        alloced = true;
      }

      mem_info info;
      info._id = id;
      info._s = s;

      cudaMemcpy(raw_mem, &info, sizeof(mem_info), cudaMemcpyHostToDevice);

      *ret = raw_mem + sizeof(mem_info);

      _available -= info._s + sizeof(mem_info);

      return alloced;
    }

    void free_mem(char* mem) {
      char* raw_mem = mem - sizeof(mem_info);

      mem_info info;
      cudaMemcpy(&info, raw_mem, sizeof(mem_info), cudaMemcpyDeviceToHost);

      deallocate(raw_mem, info._id, info._s + sizeof(mem_info));

      _available += info._s + sizeof(mem_info);
    }

    void defrag(size_t id) {
      if (!_fragemented[id]) return;

      std::vector<std::pair<char*, size_t> >& id_vector = _cache_map[id];

      if (id_vector.empty()) return;

      std::sort(id_vector.begin(), id_vector.end());

      size_t offset = 0;
      while (true) {

        size_t s = id_vector.size() + offset;

        if (s == 1)  break;

        std::pair<char*, size_t>& a = id_vector[s - 1];
        std::pair<char*, size_t>& b = id_vector[s - 2];

        if (b.first + b.second == a.first) {
          b.second += a.second;
          swap(a, id_vector.back());
          id_vector.pop_back();
        }
        else
          offset--;
      }

      _fragemented[id] = false;
    }

    void add_dist(size_t id, size_t s) {
      _dist_map[id][s]++;
    }

  protected:
    std::map<size_t, std::vector<std::pair<char*, size_t> > >  _cache_map;
    std::map<size_t, std::map<size_t, size_t> > _dist_map;
    std::map<size_t, bool> _cut_end;
    std::map<size_t, bool> _fragemented;
    std::map<char*, size_t> _ptr_map;

    void allocate(char** ret, size_t s) {
#ifdef USE_CUDNN
      CUDA_CHECK(cudaMalloc(ret, s));
#else
      *ret = new char[s];
#endif

      _ptr_map[*ret] = 0;

      _allocated += s;
      _available += s;
    }

    bool search(char** ret, size_t id, size_t s) {

      *ret = NULL;

      std::map<size_t, std::vector<std::pair<char*, size_t> > >::iterator
        itr = _cache_map.find(id);
      if (itr != _cache_map.end()) {
        std::vector<std::pair<char*, size_t> >& id_vector = itr->second;

        std::vector<std::pair<char*, size_t> >::iterator
          best_itr = id_vector.begin();
        size_t best_cost = std::numeric_limits<size_t>::max();

        for (std::vector<std::pair<char*, size_t> >::iterator itr = id_vector.begin(), e = id_vector.end(); itr != e; ++itr) {
          std::pair<char*, size_t>& entry = *itr;
          if (entry.second >= s) {
            size_t new_cost = entry.second - s;
            if (new_cost < best_cost) {
              best_cost = new_cost;
              best_itr = itr;
            }
          }

          if (best_cost == 0)
            break;
        }

        if (best_cost != std::numeric_limits<size_t>::max()) {
          if (best_cost == 0) {
            *ret = best_itr->first;
            swap(*best_itr, id_vector.back());
            id_vector.pop_back();
          }
          else {
            bool cut_end = _cut_end[id];
            if (cut_end) {
              *ret = best_itr->first + (best_itr->second - s);
            }
            else {
              *ret = best_itr->first;
              best_itr->first += s;
            }

            best_itr->second -= s;
            _cut_end[id] = !cut_end;
          }
        }
      }

      return *ret;
    }

    void deallocate(char* p, size_t id, size_t s) {
      std::vector<std::pair<char*, size_t> >& id_vector = _cache_map[id];
      id_vector.push_back(std::make_pair(p, s));

      _fragemented[id] = true;
    }
  };

  mem_mgr _mem_mgr;

void* SyncedMemory::get_cache() {
  CHECK(gpu_ptr_ == NULL);

  _cache_mtx.lock();
  _mem_mgr.defrag(device_);
  bool alloc = _mem_mgr.get_mem((char**)&gpu_ptr_, device_, size());
  if (alloc) {
    DLOG(INFO) << _mem_mgr.dump(device_);
    DLOG(INFO) << print_stacktrace();
  }

  _cache_mtx.unlock();

  //size_t free_t, total_t;
  //cudaMemGetInfo(&free_t, &total_t);

  LOG_IF(INFO, alloc) << "[LMS] memory[" << this << "] "
    << " device_=" << device_
    << " size_ = " << size()
    << " allocation=" << _mem_mgr._allocated
    << " fragmented size = " << _mem_mgr._available
    << " gpu_ptr_=" << size_t(gpu_ptr_)
    << std::flush;

  DLOG_IF(INFO, !alloc) << "[LMS] memory[" << this << "] "
    << " device_=" << device_
    << " claimed size_ = " << size()
    << " availble size = " << _mem_mgr._available
    << " gpu_ptr_=" << size_t(gpu_ptr_)
    << std::flush;

  DLOG_IF(INFO, _join) << "[LMS] memory[" << this << "] " << __FUNCTION__
    << " head_=" << head_
    << " gpu_ptr_=" << size_t(gpu_ptr_)
    << std::flush;

  return gpu_ptr_;
}

void* SyncedMemory::free_cache() {
  CHECK(gpu_ptr_);

  DLOG_IF(INFO, _join) << "memory[" << this << "] " << __FUNCTION__
    << " head_=" << head_
    << " device_=" << device_
    << " claimed size_ = " << size()
    << " availble size = " << _mem_mgr._available
    << " gpu_ptr_=" << size_t(gpu_ptr_)
    << std::flush;

  _cache_mtx.lock();
  _mem_mgr.free_mem((char*)gpu_ptr_);
  _cache_mtx.unlock();

  gpu_ptr_ = NULL;

  return NULL;
}

SyncedMemory::SyncedMemory()
  : _join(false), 
    cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
    own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
    alloc_device_(-1) {
#ifndef CPU_ONLY
#ifdef DEBUG
  CUDA_CHECK(cudaGetDevice(&device_));
#endif
#endif
}

SyncedMemory::SyncedMemory(size_t size)
  : _join(false),
    cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
    own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
    alloc_device_(-1) {
#ifndef CPU_ONLY
#ifdef DEBUG
  CUDA_CHECK(cudaGetDevice(&device_));
#endif
#endif

  CUDA_CHECK(cudaGetDevice(&device_));

  _cache_mtx.lock();
  if (_mem_mgr._demanded.size() <= device_)
    _mem_mgr._demanded.resize(device_ + 1, 0);
  _mem_mgr._demanded[device_] += size_;
  _cache_mtx.unlock();

  //assume all identical GPUs
  static size_t total_byte = 0;

  if (!total_byte) {
    size_t free_byte;
    CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte));
  }

  if (Caffe::mode() == Caffe::GPU
    &&FLAGS_lms >= 0
    && size_ >= FLAGS_lms * 1024  //in KB
    && _mem_mgr._demanded[device_] > total_byte * FLAGS_lms_frac  //total expected usage  larger than X% of total
    ) {

    CUDA_CHECK(cudaStreamCreateWithFlags(&_cache_stream, cudaStreamNonBlocking));
    to_cpu();     //force to make cpu copy
    _join = true; //big enough to join shared memory 

    _cache_mtx.lock();
    _mem_mgr.add_dist(device_, size_);
    _cache_mtx.unlock();
  }


  DLOG_IF(INFO, _join) << "memory[" << this << "] "
    << "constructed"
    << " size=" << size_
    << " join=" << std::string(_join ? "true" : "false")
    << std::flush;
}

SyncedMemory::~SyncedMemory() {
  check_device();
  if (cpu_ptr_ && own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_, alloc_device_);
  }

  if (_join&& gpu_ptr_) {
    free_cache();
  }

#ifndef CPU_ONLY
  if (gpu_ptr_ && own_gpu_data_) {
    CUDA_CHECK(cudaFree(gpu_ptr_));
  }
#endif  // CPU_ONLY
}


void SyncedMemory::push_to_cpu(bool discard) {
  DLOG_IF(INFO, _join) << "memory[" << this << "] " << __FUNCTION__ << " head_=" << head_ << std::flush;

  if (!_join)  return;

  if (discard) { //if discard, jump to synced as if memcpy is already done
    switch (head_) {
    case HEAD_AT_GPU:
    case SYNCED:
      free_cache();
      break;
    case HEAD_AT_CPU:
    case UNINITIALIZED:
      break;
    }

    head_ = UNINITIALIZED;  //repurposed for discard
  }
  else
    to_cpu();
}

inline void SyncedMemory::to_cpu() {
  check_device();

  DLOG_IF(INFO, _join) << "memory[" << this << "] " << __FUNCTION__
    << " head_=" << head_
    << " gpu_ptr_=" << size_t(gpu_ptr_)
    << std::flush;

  if (_join) {
    switch (head_) {
    case HEAD_AT_GPU:
      _mem_mgr._gpu2cpu++;
      //Minsik
      //there can be previous CUDA calls writing into gpu_ptr_, so we sync here
      //current caffe version is using default stream, but it may change in the future
      CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
      CUDA_CHECK(cudaMemcpyAsync(cpu_ptr_, gpu_ptr_, size_, cudaMemcpyDeviceToHost, _cache_stream));
      CUDA_CHECK(cudaStreamSynchronize(_cache_stream));
    case SYNCED:
      free_cache();
      head_ = HEAD_AT_CPU;
    case UNINITIALIZED:
    case HEAD_AT_CPU:
      CHECK(gpu_ptr_ == NULL);
      break;
    }

    return;
  }

  switch (head_) {
  case UNINITIALIZED:
    CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_, &alloc_device_);
    caffe_memset(size_, 0, cpu_ptr_);
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = true;
    break;
  case HEAD_AT_GPU:
#ifndef CPU_ONLY
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_, &alloc_device_);
      own_cpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
    head_ = SYNCED;
#else
    NO_GPU;
#endif
    break;
  case HEAD_AT_CPU:
  case SYNCED:
    break;
  }
}

inline void SyncedMemory::to_gpu() {
  check_device();

  DLOG_IF(INFO, _join) << "memory[" << this << "] " << __FUNCTION__
    << " head_=" << head_
    << " gpu_ptr_=" << size_t(gpu_ptr_)
    << std::flush;

  if (_join) {
    switch (head_) {
    case HEAD_AT_CPU:
      get_cache();
      _mem_mgr._cpu2gpu++;

      CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, cudaMemcpyHostToDevice, _cache_stream));
      CUDA_CHECK(cudaStreamSynchronize(_cache_stream));
      head_ = SYNCED;
      break;
    case SYNCED:
    case HEAD_AT_GPU:
      break;
    case UNINITIALIZED:
      get_cache();
      head_ = HEAD_AT_GPU;
      break;
    }
    return;
  }

#ifndef CPU_ONLY
  switch (head_) {
  case UNINITIALIZED:
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    caffe_gpu_memset(size_, 0, gpu_ptr_);
    head_ = HEAD_AT_GPU;
    own_gpu_data_ = true;
    break;
  case HEAD_AT_CPU:
    if (gpu_ptr_ == NULL) {
      CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
      own_gpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
    head_ = SYNCED;
    break;
  case HEAD_AT_GPU:
  case SYNCED:
    break;
  }
#else
  NO_GPU;
#endif
}

const void* SyncedMemory::cpu_data() {
  DLOG_IF(INFO, _join) << "memory[" << this << "] " << __FUNCTION__ << std::flush;

  check_device();
  to_cpu();
  return (const void*)cpu_ptr_;
}

void SyncedMemory::set_cpu_data(void* data) {
  DLOG_IF(INFO, _join) << "memory[" << this << "] " << __FUNCTION__ << std::flush;

  check_device();
  CHECK(data);
  if (own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_, alloc_device_);
  }
  cpu_ptr_ = data;
  head_ = HEAD_AT_CPU;
  own_cpu_data_ = false;
}

const void* SyncedMemory::gpu_data() {
  DLOG_IF(INFO, _join) << "memory[" << this << "] " << __FUNCTION__ << std::flush;

  check_device();
#ifndef CPU_ONLY
  to_gpu();
  return (const void*)gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

void SyncedMemory::set_gpu_data(void* data) {
  check_device();

  if (_join) {
    DLOG(INFO) << "[LMS] memory[" << this << "] " << __FUNCTION__
      << " size_=" << size_
      << " gpu_ptr_=" << size_t(gpu_ptr_) << std::flush;

    if (gpu_ptr_)
      free_cache();

    _join = false;
  }

#ifndef CPU_ONLY
  CHECK(data);
  if (own_gpu_data_) {
    CUDA_CHECK(cudaFree(gpu_ptr_));
  }
  gpu_ptr_ = data;
  head_ = HEAD_AT_GPU;
  own_gpu_data_ = false;
#else
  NO_GPU;
#endif
}

void* SyncedMemory::mutable_cpu_data() {
  DLOG_IF(INFO, _join) << "memory[" << this << "] " << __FUNCTION__
    << " gpu_ptr_=" << size_t(gpu_ptr_)
    << std::flush;

  check_device();
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}

void* SyncedMemory::mutable_gpu_data() {
  DLOG_IF(INFO, _join) << "memory[" << this << "] " << __FUNCTION__ << std::flush;

  check_device();
#ifndef CPU_ONLY
  to_gpu();
  head_ = HEAD_AT_GPU;
  return gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

#ifndef CPU_ONLY
void SyncedMemory::async_gpu_push(const cudaStream_t& stream) {
  check_device();

  if (_join) {
    CHECK(head_ == HEAD_AT_CPU);
    CHECK(gpu_ptr_ == NULL);
    get_cache();

    DLOG(INFO) << "memory[" << this << "] " << __FUNCTION__
      << " size_=" << size_
      << " gpu_ptr_=" << size_t(gpu_ptr_)
      << " cpu_ptr_=" << cpu_ptr_
      << std::flush;

    CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, cudaMemcpyHostToDevice, stream));

    head_ = SYNCED;
    return;
  }

  CHECK(head_ == HEAD_AT_CPU);
  if (gpu_ptr_ == NULL) {
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    own_gpu_data_ = true;
  }
  const cudaMemcpyKind put = cudaMemcpyHostToDevice;
  CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
  // Assume caller will synchronize on the stream before use
  head_ = SYNCED;
}
#endif

void SyncedMemory::check_device() {
#ifndef CPU_ONLY
#ifdef DEBUG
  int device;
  cudaGetDevice(&device);
  CHECK(device == device_);
  if (gpu_ptr_ && own_gpu_data_) {
    cudaPointerAttributes attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, gpu_ptr_));
    CHECK(attributes.device == device_);
  }
#endif
#endif
}

}  // namespace caffe

