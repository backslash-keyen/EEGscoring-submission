# Task 2e: why the lower number is the one to report

Our first sleep-staging program reported 85% accuracy. The corrected version scores about 83%, and the lower figure is the honest one.

The first test was too easy. Each recording runs for almost a whole day, and for two-thirds of it the person is awake. A program that always answered "awake" would score 69%. The corrected test keeps only the night, plus half an hour either side.

The old test had two other flaws. It included people the program had already studied, since their other night was in its training material. And its best version was picked using the test itself. Real patients will be new to it.

A fairer score gives every sleep stage equal weight, so easy wakefulness earns nothing extra. On that score the corrected program improved from 0.63 to 0.79 (out of 1). The headline dropped because the test got harder, while the program got better at telling sleep stages apart.

<!-- Sources: given script seed 42 (task2/baseline/2a_baseline_seed42.txt: accuracy 0.847, macro-F1 0.628),
     fixed pipeline, mean of seeds 42-44 (task2/ledger_table.md FIXED row: accuracy 0.827, macro-F1 0.788),
     'always Wake' 68.7% (task2/evidence/d10_wake_padding_result.txt). Body: 154 words. -->
