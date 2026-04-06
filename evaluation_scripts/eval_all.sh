# TEST_DATA_PATHs=("ReasonSeg_test" "ReasonSeg_val" "refcoco_testA" $2)
TEST_DATA_PATHs=("ReasonSeg_test" "ReasonSeg_val" "refcoco_testA" "refcocoplus_testA" "refcocog_test" $2)

CHECKPOINT_PATH=$1

for test_data in "${TEST_DATA_PATHs[@]}"; do
    echo "============> Evaluation $test_data ...."
    sleep 5s
    bash evaluation_scripts/eval_one.sh $CHECKPOINT_PATH $test_data $3
    sleep 10s
done


echo ""
echo "============>All evaluatin resutls"
cat $CHECKPOINT_PATH/reasonseg_eval_results/eval.json