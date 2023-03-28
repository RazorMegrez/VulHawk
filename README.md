# VulHawk
This is the official repository for VulHawk.

## Get Started
### Prerequisites
- Windows (MacOS and Linux also work)
- Python 3.8.2 (64 bit)
- PyTorch 1.13.1
- CUDA 11.7
- IDA pro 7.5+ (only used for dataset processing)

### Quick Start
- File Environment Identification
```
python 2_FileEnvironmentIdentification.py
```
- Function Embedding Generation
```
python 3_function_embedding_generation.py
```
- Binary Function Similarity Detection
```
python 4_binary_code_search.py
```

---

### Entropy Figure
#### x86-O0

![](figure/ent_pic/O0/clang-10_coreutils-8.30_base64.png)
![](figure/ent_pic/O0/clang-10_coreutils-8.30_expr.png)
![](figure/ent_pic/O0/clang-10_sqlite-autoconf-3370100_libsqlite3.so.png)
![](figure/ent_pic/O0/clang-10_curl-7.80.0_libcurl.so.png)
![](figure/ent_pic/O0/clang-10_curl-7.80.0_curl.png)
![](figure/ent_pic/O0/clang-10_putty-0.74_plink.png)
![](figure/ent_pic/O0/clang-10_putty-0.74_puttygen.png)
![](figure/ent_pic/O0/clang-10_libmicrohttpd-0.9.75_libmicrohttpd.so.png)

#### arm-O1

![](figure/ent_pic/O1/arm-64_wget2-2.0.0_libwget.so.png) 
![](figure/ent_pic/O1/arm-64_coreutils-8.30_expr.png)
![](figure/ent_pic/O1/arm-64_coreutils-8.30_id.png)
![](figure/ent_pic/O1/arm-64_curl-7.80.0_libcurl.so.png)
![](figure/ent_pic/O1/arm-64_curl-7.80.0_curl.png)
![](figure/ent_pic/O1/arm-64_sqlite-autoconf-3370100_sqlite3.png)
![](figure/ent_pic/O1/arm-64_putty-0.74_puttygen.png)
![](figure/ent_pic/O1/arm-64_libmicrohttpd-0.9.75_libmicrohttpd.so.png)

#### x86-O3

![](figure/ent_pic/O3/x86-64_coreutils-8.30_expr.png)
![](figure/ent_pic/O3/x86-64_sqlite-autoconf-3370100_sqlite3.png)
![](figure/ent_pic/O3/x86-64_coreutils-8.30_dir.png)
![](figure/ent_pic/O3/x86-64_wget2-2.0.0_libwget_xml.so.png)
![](figure/ent_pic/O3/x86-64_curl-7.80.0_curl.png)
![](figure/ent_pic/O3/x86-64_putty-0.74_plink.png)
![](figure/ent_pic/O3/x86-64_putty-0.74_puttygen.png)
![](figure/ent_pic/O3/x86-64_libmicrohttpd-0.9.75_libmicrohttpd.so.png)