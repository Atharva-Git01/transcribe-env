# Screening set v1: latest result per candidate

| candidate | device | content CER | strict CER | S recall | S switched | R content CER | R words not in truth | N neg. wrong | N num. wrong | G content CER | RTF | load s | GPU MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| indic_transcribe | cuda bfloat16 | 0.2069 | 0.2437 | 0.0 | 0/241 | 0.3861 | 74 | 10/63 | 2/36 | 0.2723 | 0.33 | 40.8 | 2665 |
| transfer_it_lora_v1_test_call | cuda spelling transfer | 0.2069 | 0.244 | 0.0408 | 7/241 | 0.3861 | 74 | 10/63 | 2/36 | 0.2723 | 0.716 | 81.3 | 2665 |
| transfer_it_lora_v2_test_call | cuda spelling transfer | 0.2069 | 0.2448 | 0.0204 | 7/241 | 0.3861 | 74 | 10/63 | 2/36 | 0.2723 | 0.653 | 131.8 | 2665 |
| it_lora_v1_test_call | cuda bfloat16 | 0.2111 | 0.2423 | 0.0465 | 7/241 | 0.3648 | 77 | 20/63 | 3/36 | 0.2857 | 0.386 | 40.5 | 2665 |
| it_lora_v2_test_call | cuda bfloat16 | 0.2111 | 0.242 | 0.0444 | 7/241 | 0.3843 | 74 | 19/63 | 3/36 | 0.2991 | 0.323 | 91.0 | 2665 |
| indic_conformer_ctc | cpu onnx fp32 (ctc) | 0.2466 | 0.2771 | 0.0714 | 1/241 | 0.411 | 84 | 13/63 | 4/36 | 0.3705 | 0.334 | 87.8 | 0 |
| indic_conformer_rnnt | cpu onnx fp32 (rnnt) | 0.2581 | 0.2911 | 0.0732 | 1/241 | 0.427 | 65 | 26/63 | 2/36 | 0.3795 | 0.69 | 26.6 | 0 |
| indicwhisper_mr | cuda int8_float16 | 0.4074 | 0.4265 | 0.0345 | 0/241 | 0.806 | 141 | 33/63 | 17/36 | 0.4732 | 0.489 | 9.7 | 1915 |
| mms_1b_all_mar | cuda float16 | 0.4327 | 0.4538 | 0.125 | 6/241 | 0.831 | 79 | 51/63 | 35/36 | 0.5513 | 0.034 | 129.8 | 3097 |
| whisper_large_v3_as_spoken_prompt | cuda int8_float16 | 0.5825 | 0.6163 | 0.45 | 75/241 | 1.4662 | 234 | 55/63 | 33/36 | 0.6451 | 0.635 | 142.6 | 2779 |
| phoneme_xlsr53_espeak (judge) | cuda float16 | — | — | 0.38 | 114/241 | — | — | — | — | — | 0.016 | 21.4 | 1649 |
| phoneme_xlsr53_espeak_ctc (judge) | cuda float16 | — | — | 0.68 | 160/241 | — | — | — | — | — | 0.01 | 24.9 | 1681 |
