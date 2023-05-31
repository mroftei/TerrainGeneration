# Automatic Modulation Recognition Model Zoo

## Datasets
### RadioML 2016.04c
11 modulations
- Digital: BPSK, QPSK, 8PSK, 16QAM, 64QAM, BFSK, CPFSK, and PAM4
- Analog: WB-FM, AM-SSB, and AM-DSB

Channel Effects
- Thermal noise is AWGN
- Symbol timing offset, sample rate offset, carrier frequency offset and phase difference
- Temporal shifting, scaling, linear mixing/rotating between channels, and spinning
- Random filtering
- multi-path fading or frequency selective fading

[Download](https://www.deepsig.ai/datasets)

T. J. O’Shea, J. Corgan, and T. C. Clancy, “Convolutional Radio Modulation Recognition Networks,” Communications in Computer and Information Science, vol. 629, pp. 213–226, Feb. 2016.

### RadioML 2016.10b
Same as 2016.04c with more samples.

[Download](https://www.deepsig.ai/datasets)

T. J. O’Shea and N. West, “Radio Machine Learning Dataset Generation with GNU Radio,” Proceedings of the GNU Radio Conference, vol. 1, no. 1, Art. no. 1, Sep. 2016, Accessed: Apr. 26, 2023. [Online]. Available: https://pubs.gnuradio.org/index.php/grcon/article/view/11

### RadioML 2018.01a
24 modulations (11 for easy version)
- Digital: OOK, 4ASK, 8ASK, BPSK, QPSK, 8PSK, 16PSK, 32PSK, 16APSK, 32APSK, 64APSK, 128APSK, 16QAM, 32QAM, 64QAM, 128QAM, 256QAM, GMSK, OQPSK
- Analog: AM-SSB-WC, AM-SSB-SC, AM-DSB-WC, AM-DSB-SC, FM

Channel Effects: same as 2016 dataset

[Download](https://www.deepsig.ai/datasets)

T. J. O’Shea, T. Roy, and T. C. Clancy, “Over the Air Deep Learning Based Radio Signal Classification,” IEEE Journal on Selected Topics in Signal Processing, vol. 12, no. 1, pp. 168–179, Dec. 2017, doi: 10.1109/jstsp.2018.2797022.

### HisarMod2019.1
Matlab genrated dataset with better doumented channel effects

26 modulations
- Analog: AM–DSB, AM–SC, AM–USB, AM–LSB, FM, PM
- FSK: 2–FSK, 4–FSK, 8–FSK, 16–FSK
- Pulse amplitude modulation (PAM): 4–PAM, 8–PAM, 16–PAM
- PSK: BPSK. QPSK, 8–PSK, 16–PSK, 32–PSK, 64–PSK
- Quadrature amplitude modulation (QAM): 4–QAM, 8–QAM, 16–QAM, 32–QAM, 64–QAM, 128–QAM, 256–QAM

Channel Impairments (equally distributed across Modulation/SNR combos (300/combo)): 
- Ideal: no fading, AWGN
- Static: channel coefficients constant
- Rayleigh: non-LOS conditions
- Rician: k=3 for mild fading
- Nakagami: m=2
- Multipath channel taps are equally likely in \[4,6\] as per ITU–R M1225

[Download](https://ieee-dataport.org/open-access/hisarmod-new-challenging-modulated-signals-dataset)

K. Tekbıyık, A. R. Ekti, A. Görçin, G. K. Kurt, and C. Keçeci, “Robust and Fast Automatic Modulation Classification with CNN under Multipath Fading Channels,” in 2020 IEEE 91st Vehicular Technology Conference (VTC2020-Spring), May 2020, pp. 1–6. doi: 10.1109/VTC2020-Spring48590.2020.9128408.


## Models

### SimpleConv
Simple convolutional model with 5 blocks of 2 Convolutions with Max Pooling after each block. Classifier is a 3-layer MLP.

N. Soltani, K. Sankhe, S. Ioannidis, D. Jaisinghani, and K. Chowdhury, “Spectrum Awareness at the Edge: Modulation Classification using Smartphones,” 2019 IEEE International Symposium on Dynamic Spectrum Access Networks, DySPAN 2019, Nov. 2019, doi: 10.1109/DYSPAN.2019.8935775.

### CLDNN
Convolutional Feature Extractor followed by a GRU. 2-layer MLP classifier.

N. E. West and T. O’Shea, “Deep architectures for modulation recognition,” in 2017 IEEE International Symposium on Dynamic Spectrum Access Networks (DySPAN), Mar. 2017, pp. 1–6. doi: 10.1109/DySPAN.2017.7920754.

https://github.com/brysef/rfml/blob/master/rfml/nn/model/cldnn.py

### ResNet
