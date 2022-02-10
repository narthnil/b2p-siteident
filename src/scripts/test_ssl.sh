for dir in ./results/ssl-v1/*/*/     # list directories in the form "/tmp/dirname/"
do
    dir=${dir%*/}      # remove the trailing "/"
    echo "${dir}"    # print everything after the final "/"
    python test_ssl.py --save_dir ${dir}
done
for dir in ./results/ssl-v2/*/*/     # list directories in the form "/tmp/dirname/"
do
    dir=${dir%*/}      # remove the trailing "/"
    echo "${dir}"    # print everything after the final "/"
    python test_ssl.py --save_dir ${dir}
done
