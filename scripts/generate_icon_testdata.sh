#!/bin/bash

PROBTEST_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. &> /dev/null && pwd )"

if [[ -z "$ICON_DATA" ]]; then
    echo "Path to the collected ICON run and output data must be set as 'ICON_DATA'"
    exit 1
fi

if [[ -z "$PROBTEST_DATA" ]]; then
    echo "Path where the output of probtest should be stored must be set as 'PROBTEST_DATA'"
    exit 1
fi

for exp in atm_amip_test mch_opr_r04b07_lhn_12
do
    if [[ "$exp" == "atm_amip_test" ]]; then
        file_ids=atm_3d,lnd
    fi
    if [[ "$exp" == "mch_opr_r04b07_lhn_12" ]]; then
        file_ids=atm_3d_ml,atm_3d_hl,atm_3d_pl
    fi
    python $PROBTEST_DIR/probtest.py init \
        --codebase-install $ICON_DATA \
        --experiment-name $exp \
        --file-ids $file_ids \
        --reference $PROBTEST_DATA \
        --template-name $PROBTEST_DIR/templates/testdata.jinja \
        --member_ids 1,2 \
        --time_dim time \
        --timing-current $ICON_DATA/probtest_testdata \
        --timing-reference $ICON_DATA/performance_reference/probtest_testdata \
        || exit 1

    python $PROBTEST_DIR/probtest.py run-ensemble || exit 1
    python $PROBTEST_DIR/probtest.py stats || exit 1
    python $PROBTEST_DIR/probtest.py tolerance || exit 1
    python $PROBTEST_DIR/probtest.py cdo-table || exit 1
    python $PROBTEST_DIR/probtest.py performance || exit 1
    python $PROBTEST_DIR/probtest.py check-plot || exit 1
    python $PROBTEST_DIR/probtest.py perturb || exit 1

    # Use the `true` tool to test the non-dry code path. `true` does nothing and returns 0. See `man true`.
    python $PROBTEST_DIR/probtest.py run-ensemble --no-dry --submit-command=true || exit 1
done

# the following make use of long-running performance database for mch_ch_lowres
# python $PROBTEST_DIR/probtest.py performance-plot || exit 1
# python $PROBTEST_DIR/probtest.py performance-meta-data || exit 1
