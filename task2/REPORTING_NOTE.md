# Task 2e: why the lower number is the one to report

The first version of our sleep-staging program reported 85% accuracy; the corrected version scores about 83%. The lower figure is the honest one.

The first test was too easy. Each recording runs for almost a whole day, and two-thirds of it is the person awake with the lights on. A program that always answered "awake" would already score 69%. The corrected test keeps only the night, plus half an hour either side.

The first test also included people the program had already studied (their other night was in its training material), and it picked its best version by peeking at the test. Real patients will be new to it.

On a balanced score that counts every sleep stage equally, rather than rewarding easy wakefulness, the corrected program improved from 0.63 to 0.79 (out of 1). It looks weaker only where the old test flattered it.

<!-- Sources: given script seed 42 (task2/baseline/2a_baseline_seed42.txt: accuracy 0.847, macro-F1 0.628);
     fixed pipeline, mean of seeds 42-44 (task2/ledger_table.md FIXED row: accuracy 0.827, macro-F1 0.788);
     'always Wake' 68.7% (task2/evidence/d10_wake_padding_result.txt). Body: 145 words. -->
