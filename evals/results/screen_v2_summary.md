# Screening set v2: latest result per candidate

| candidate | device | content CER | strict CER | S recall | S switched | R content CER | R words not in truth | N neg. wrong | N num. wrong | G content CER | RTF | load s | GPU MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| indic_transcribe | cuda bfloat16 | 0.1749 | 0.2007 | 0.0 | 0/424 | 0.3758 | 106 | 16/103 | 9/60 | 0.2819 | 0.505 | 74.1 | 1705 |
| transfer_it_lora_v3_test_call | cuda spelling transfer | 0.1749 | 0.2153 | 0.2963 | 121/424 | 0.3758 | 106 | 16/103 | 9/60 | 0.2819 | 0.972 | 125.0 | 2773 |
| it_lora_v3_test_call | cuda bfloat16 | 0.1899 | 0.2263 | 0.4 | 121/424 | 0.3346 | 97 | 30/103 | 12/60 | 0.2703 | 0.467 | 50.9 | 2773 |
