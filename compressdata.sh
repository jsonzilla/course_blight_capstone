# compress all files in the data directory separately with 7z ultra compression
cd data
for file in *; do
    7z a -t7z -m0=lzma2 -mx=9 -mfb=64 -md=32m -ms=on "$file".7z "$file"
done