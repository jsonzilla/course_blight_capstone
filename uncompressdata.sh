# uncompress all files in the data directory separately with 7z
# move the file to the data directory
# delete the temporary folder
cd data
for file in *; do
    7z x "$file" -o../temp
    mv ../temp/* .
    rm -r ../temp
done