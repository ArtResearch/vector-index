# syntax=docker/dockerfile:1

# Builder with GCC 15 (preferred). If unavailable in your registry, switch to gcc:14-bookworm.
FROM gcc:15-bookworm AS builder

ARG BOOST_VERSION=1.86.0
ARG BOOST_DIR=boost_1_86_0

# Tools for fetching/building Boost
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      wget ca-certificates bzip2 cmake make libssl-dev libcurl4-openssl-dev && \
    rm -rf /var/lib/apt/lists/*

# Build and install only Boost.Program_options (Boost.URL is header-only)
WORKDIR /tmp
RUN wget -q https://archives.boost.io/release/${BOOST_VERSION}/source/${BOOST_DIR}.tar.bz2 && \
    tar -xf ${BOOST_DIR}.tar.bz2 && \
    cd ${BOOST_DIR} && \
    ./bootstrap.sh --with-libraries=program_options,system,url && \
    ./b2 -j$(nproc) cxxstd=20 link=shared variant=release install

# Copy project sources (context is ./metadata)
WORKDIR /src
COPY . /src

# Build the server with CMake (uses metadata/CMakeLists.txt)
RUN mkdir -p /out && \
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j"$(nproc)" && \
    install -D -m 0755 build/bin/searchUSearchIndex_server /out/search_usearch_server

# Runtime image
FROM gcc:15-bookworm AS runtime

# Minimal runtime deps and CA roots
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates libstdc++6 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy binary and the single Boost shared library it needs
COPY --from=builder /out/search_usearch_server /usr/local/bin/search_usearch_server
COPY --from=builder /usr/local/lib/libboost_program_options*.so* /usr/local/lib/
COPY --from=builder /usr/local/lib/libboost_system*.so* /usr/local/lib/
COPY --from=builder /usr/local/lib/libboost_url*.so* /usr/local/lib/
COPY --from=builder /src/build/third_party/opencv/lib/libopencv_*.so* /usr/local/lib/

ENV LD_LIBRARY_PATH=/usr/local/lib
EXPOSE 8545

# Default entrypoint. docker-compose can pass -i/-m/-p/-e as command args.
ENTRYPOINT ["search_usearch_server"]
CMD ["--help"]
