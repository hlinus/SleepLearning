#!/usr/bin/env bash

# download some sleep-edf data
mkdir -p sleepedf
wget -P sleepedf https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4001E0-PSG.edf
wget -P sleepedf https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4001E0-PSG.edf.hyp
wget -P sleepedf https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4001EC-Hypnogram.edf
wget -P sleepedf https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4002E0-PSG.edf
wget -P sleepedf https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4002E0-PSG.edf.hyp
wget -P sleepedf https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4002EC-Hypnogram.edf

# download some physionet18 data
mkdir -p physionet-challenge-train
wget -r -nH --cut-dirs=4 --no-parent --reject="index.html*" -P physionet-challenge-train https://physionet.org/pn3/challenge/2018/training/tr04-0362/
wget -r -nH --cut-dirs=4 --no-parent --reject="index.html*" -P physionet-challenge-train https://physionet.org/pn3/challenge/2018/training/tr11-0064/
wget -r -nH --cut-dirs=4 --no-parent --reject="index.html*" -P physionet-challenge-train https://physionet.org/pn3/challenge/2018/training/tr05-0838/
wget -r -nH --cut-dirs=4 --no-parent --reject="index.html*" -P physionet-challenge-train https://physionet.org/pn3/challenge/2018/training/tr14-0016/