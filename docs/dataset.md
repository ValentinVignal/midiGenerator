# midiGenerator
Generate midi file with deep neural network :notes:

# Dataset 

## One note Dataset

| Dataset | Can use `mono` :musical_note: |
| :---: | :---: |
| Bach(...) | :heavy_check_mark: |
| SimpleDataset | :x: |

## Informations
### Simple Dataset

| Name | Number of songs | Piano | Trombone | Flute | Violin | All | Mono |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **(1)** | 1 | 43:54 | 20:35 | 46:59 | 43:54 | 20:59 | :x: |
| **2** | 3 | 42:56 | 20:44 | 46:63 | 36:54 | 20:63 | :x: |


### BachChorale

Use the option `--bach`

| | nb songs | nb accepted songs :heavy_check_mark: | notes range :musical_note: | nb measures :musical_score: | Mono |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Small** | 3 | 2 | 22:54 | 23 | :heavy_check_mark: |
| **Medium** | 15 | 11 | 19:58 | 155 | :heavy_check_mark: |
| **Medium2** | 30 | 21 | 18:61 | 322 | :heavy_check_mark: |
| **Big** | 100 | 74 | 17:61 | 1127 | :heavy_check_mark: |

# Other Dataset

| Name | nb songs | instruments | nb accepted songs  :heavy_check_mark: | notes range :musical_note: | nb measures :musical_score: | Mono |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scale** | 1 | Trombone, Flute | 1 | 27:52 | 32 | :heavy_check_mark: |
| **MyDatasetMono** | 9 | Flute, Piano, Trombone, Bass | 9 | 7:75 | 578 | :heavy_check_mark: 


