#!/bin/bash
# Monitor script — runs on the B200 server.
# Checks experiment progress every 30 min. When free_explore is done,
# pushes results and starts scenario experiments.
#
# Usage: nohup bash monitor.sh > /workspace/monitor.log 2>&1 &

cd /workspace/repo

while true; do
    echo ""
    echo "=== MONITOR CHECK $(date) ==="

    # Count completed result files
    N_RESULTS=$(ls results/self_steer_v2_*.json 2>/dev/null | wc -l)
    echo "Result files: $N_RESULTS"

    # Check if run_experiments.py is still running
    RUNNING=$(ps aux | grep run_experiments | grep -v grep | wc -l)
    STEER_RUNNING=$(ps aux | grep self_steer_v2 | grep -v grep | wc -l)
    echo "run_experiments.py running: $RUNNING"
    echo "self_steer_v2.py running: $STEER_RUNNING"

    # If nothing is running and we have results, push and start next batch
    if [ "$RUNNING" -eq 0 ] && [ "$STEER_RUNNING" -eq 0 ] && [ "$N_RESULTS" -gt 0 ]; then
        echo ">>> Experiments finished! Pushing results..."

        # Git push results
        git add results/*.json
        git commit -m "Experiment results: $N_RESULTS files (auto-push from monitor)"
        git push

        echo ">>> Results pushed."

        # Check what's been done
        N_FREE=$(ls results/self_steer_v2_research_*.json results/self_steer_v2_other_model_*.json results/self_steer_v2_potions_*.json results/self_steer_v2_minimal_*.json results/self_steer_v2_no_tools_*.json results/self_steer_v2_full_technical_*.json 2>/dev/null | wc -l)
        N_SCEN_A=$(ls results/self_steer_v2_*scen_a_*.json 2>/dev/null | wc -l)
        N_SCEN_C=$(ls results/self_steer_v2_*scen_c_*.json 2>/dev/null | wc -l)
        N_SCEN_F=$(ls results/self_steer_v2_*scen_f_*.json 2>/dev/null | wc -l)

        echo "Free exploration: $N_FREE files"
        echo "Scenario A: $N_SCEN_A files"
        echo "Scenario C: $N_SCEN_C files"
        echo "Scenario F: $N_SCEN_F files"

        # If free exploration done but scenarios not started, run scenarios
        if [ "$N_FREE" -ge 40 ] && [ "$N_SCEN_A" -eq 0 ]; then
            echo ">>> Free exploration done. Starting Scenario A (interference)..."
            MODEL_PATH=/workspace/llama-3.3-70b SAE_PATH=/workspace/sae-l50 LABELS_PATH=/workspace/feature_labels.json \
                nohup python run_experiments.py --selfhost http://localhost:8000 --group scenario_a --temp 0.7 \
                > /workspace/scenario_a.log 2>&1 &
            echo ">>> Scenario A started (PID: $!)"

        elif [ "$N_SCEN_A" -ge 30 ] && [ "$N_SCEN_C" -eq 0 ]; then
            echo ">>> Scenario A done. Starting Scenario C (wireheading)..."
            MODEL_PATH=/workspace/llama-3.3-70b SAE_PATH=/workspace/sae-l50 LABELS_PATH=/workspace/feature_labels.json \
                nohup python run_experiments.py --selfhost http://localhost:8000 --group scenario_c --temp 0.7 \
                > /workspace/scenario_c.log 2>&1 &
            echo ">>> Scenario C started (PID: $!)"

        elif [ "$N_SCEN_C" -ge 20 ] && [ "$N_SCEN_F" -eq 0 ]; then
            echo ">>> Scenario C done. Starting Scenario F (observation)..."
            MODEL_PATH=/workspace/llama-3.3-70b SAE_PATH=/workspace/sae-l50 LABELS_PATH=/workspace/feature_labels.json \
                nohup python run_experiments.py --selfhost http://localhost:8000 --group scenario_f --temp 0.7 \
                > /workspace/scenario_f.log 2>&1 &
            echo ">>> Scenario F started (PID: $!)"

        elif [ "$N_SCEN_F" -ge 10 ]; then
            echo ">>> ALL EXPERIMENTS COMPLETE."
            echo ">>> Final push..."
            git add results/*.json
            git commit -m "All experiments complete: $N_RESULTS total files"
            git push
            echo ">>> Done. Monitor exiting."
            exit 0
        fi
    fi

    echo "Sleeping 30 min..."
    sleep 1800
done
