#!/bin/bash

datasets=("data/datasets/Hotel_Reviews.csv" "data/datasets/blog_authorship_corpus.csv")
configs_all=("data/configurations/e_hotel_reviews.yaml" "data/configurations/e_blog_authorship_corpus.yaml")
configs_gpe=("data/configurations/e_hotel_reviews_gpe.yaml" "data/configurations/e_blog_authorship_corpus_gpe.yaml")
weights=(1 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0)

TOKEN="1306410521:AAGPIP2dm6wBsMsZVscgJwr7VdwO8vP8yuQ"
ID="659247640"
URL="https://api.telegram.org/bot$TOKEN/sendMessage"

cd ~
source anon_env/bin/activate
cd 2020ss-thesis-fabian
log_dir="../logs"
mkdir -p "$log_dir"
run_parallel=true
timeout=1
n_proc=9

curl -s -X POST $URL -d chat_id=$ID -d text="------- Running experiments -------" > /dev/null 2>&1

for ((idx=0; idx<${#datasets[@]}; ++idx)); do
    dataset=${datasets[idx]}
    config=${configs_all[idx]}
    ds=$(basename -- "$dataset")
    ds="${ds%.*}"
    result_dir="${ds}_all_entities"
    for weight in "${weights[@]}"
    do
        must_be_scheduled=true
        cmd="python anon/experiment_runner.py -i $dataset -c $config -r $result_dir -w $weight &> ${log_dir}/${result_dir}_mondrian-${weight}.log"
        if [ "$run_parallel" = true ]
        then
            while [ "$must_be_scheduled" = true ]; do
                if [ "`ps auxw | grep "python anon/experiment_runner.py" | wc -l`" -lt $n_proc ]; then
                    must_be_scheduled=false
                    echo "starting a new anon instance"
                    #echo $cmd
                    (eval $cmd; curl -s -X POST $URL -d chat_id=$ID -d text="${result_dir}_mondrian-${weight} finished" > /dev/null 2>&1) &
                fi
                echo "`ps auxw | grep "python anon/experiment_runner.py" | wc -l` anon instance(s) running"
                sleep $timeout
            done
        else
            (eval $cmd; curl -s -X POST $URL -d chat_id=$ID -d text="${result_dir}_mondrian-${weight} finished" > /dev/null 2>&1)
        fi
    done
    # gdf strategy
    must_be_scheduled=true
    cmd="python anon/experiment_runner.py -i $dataset -c $config -r $result_dir &> ${log_dir}/${result_dir}_gdf.log"
    if [ "$run_parallel" = true ]
    then
        while [ "$must_be_scheduled" = true ]; do
            if [ "`ps auxw | grep "python anon/experiment_runner.py" | wc -l`" -lt $n_proc ]; then
                must_be_scheduled=false
                echo "starting a new anon instance"
                #echo $cmd
                (eval $cmd; curl -s -X POST $URL -d chat_id=$ID -d text="${result_dir}_gdf finished" > /dev/null 2>&1) &
            fi
            echo "`ps auxw | grep "python anon/experiment_runner.py" | wc -l` anon instance(s) running"
            sleep $timeout
        done
    else
        (eval $cmd; curl -s -X POST $URL -d chat_id=$ID -d text="${result_dir}_gdf finished" > /dev/null 2>&1)
    fi
done

for ((idx=0; idx<${#datasets[@]}; ++idx)); do
    dataset=${datasets[idx]}
    config=${configs_gpe[idx]}
    ds=$(basename -- "$dataset")
    ds="${ds%.*}"
    result_dir="${ds}_gpe"
    for weight in "${weights[@]}"
    do
        must_be_scheduled=true
        cmd="python anon/experiment_runner.py -i $dataset -c $config -r $result_dir -w $weight &> ${log_dir}/${result_dir}_mondrian-${weight}.log"
        if [ "$run_parallel" = true ]
        then
            while [ "$must_be_scheduled" = true ]; do
                if [ "`ps auxw | grep "python anon/experiment_runner.py" | wc -l`" -lt $n_proc ]; then
                    must_be_scheduled=false
                    echo "starting a new anon instance"
                    #echo $cmd
                    (eval $cmd; curl -s -X POST $URL -d chat_id=$ID -d text="${result_dir}_mondrian-${weight} finished" > /dev/null 2>&1) &
                fi
                echo "`ps auxw | grep "python anon/experiment_runner.py" | wc -l` anon instance(s) running"
                sleep $timeout
            done
        else
            (eval $cmd; curl -s -X POST $URL -d chat_id=$ID -d text="${result_dir}_mondrian-${weight} finished" > /dev/null 2>&1)
        fi
    done
    # gdf strategy
    must_be_scheduled=true
    cmd="python anon/experiment_runner.py -i $dataset -c $config -r $result_dir &> ${log_dir}/${result_dir}_gdf.log"
    if [ "$run_parallel" = true ]
    then
        while [ "$must_be_scheduled" = true ]; do
            if [ "`ps auxw | grep "python anon/experiment_runner.py" | wc -l`" -lt $n_proc ]; then
                must_be_scheduled=false
                echo "starting a new anon instance"
                #echo $cmd
                (eval $cmd; curl -s -X POST $URL -d chat_id=$ID -d text="${result_dir}_gdf finished" > /dev/null 2>&1) &
            fi
            echo "`ps auxw | grep "python anon/experiment_runner.py" | wc -l` anon instance(s) running"
            sleep $timeout
        done
    else
        (eval $cmd; curl -s -X POST $URL -d chat_id=$ID -d text="${result_dir}_gdf finished" > /dev/null 2>&1)
    fi
done

wait

curl -s -X POST $URL -d chat_id=$ID -d text="------- Experiments finished -------" > /dev/null 2>&1
