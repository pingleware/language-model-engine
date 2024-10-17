rm -frd build
cmake -DCMAKE_OSX_ARCHITECTURES=arm64 -B build .
cd build
make
cd ..