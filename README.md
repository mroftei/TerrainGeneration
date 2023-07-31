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
Model Performance on CSPB2018 v2 test dataset 
| Dataset   | 1 chan | 6 chan |
|-----------|--------|--------|
| CNN1      |  | 0.7366 |
| CNN2      |  |  |
| MCNET     |  |  |
| IC-AMCNET |  |  |
| ResNet    |  |  |
| DenseNet  |  |  |
| GRU       |  | 0.7337 |
| LSTM      | 0.6744 | 0.7566 |
| DAE       |  |  |
| MCLDNN    |  | 0.7331 |
| CLDNN     |  |  |
| CLDNN2    |  |  |
| CGDNet    |  |  |
| PET-CGDNN |  |  |
### ConvBlocks
Convolutional model with 5 blocks of 2 Convolutions with Max Pooling after each block. Classifier is a 3-layer MLP.

N. Soltani, K. Sankhe, S. Ioannidis, D. Jaisinghani, and K. Chowdhury, “Spectrum Awareness at the Edge: Modulation Classification using Smartphones,” 2019 IEEE International Symposium on Dynamic Spectrum Access Networks, DySPAN 2019, Nov. 2019, doi: 10.1109/DYSPAN.2019.8935775.

### CLDNN
Convolutional Feature Extractor followed by a GRU. 2-layer MLP classifier.

N. E. West and T. O’Shea, “Deep architectures for modulation recognition,” in 2017 IEEE International Symposium on Dynamic Spectrum Access Networks (DySPAN), Mar. 2017, pp. 1–6. doi: 10.1109/DySPAN.2017.7920754.

https://github.com/brysef/rfml/blob/master/rfml/nn/model/cldnn.py

### CLDNN2
Convolutional Long Deep Neural Network (CNN + GRU + MLP)

N. E. West and T. O'Shea, “Deep architectures for modulation recognition,” in IEEE International Symposium on Dynamic Spectrum Access Networks (DySPAN), pp. 1-6, IEEE, 2017.

### CNN1
Variation of VTCNN2

T. J. O'Shea, J. Corgan, and T. C. Clancy, “Convolutional radio modulation recognition networks,” in International Conference on Engineering Applications of Neural Networks, pp. 213-226, Springer,2016.

S. C. Hauser, W. C. Headley, and A. J.  Michaels, “Signal detection effects ondeep neural networks utilizing raw iq for modulation classification,” in Military Communications Conference, pp. 121-127, IEEE, 2017.

### CNN2
Convs with 2-layer MLP

K. Tekbiyik, A. R. Ekti, A. Görçin, G. K. Kurt, and C. Keçeci, “Robust and Fast Automatic Modulation Classification with CNN under Multipath Fading Channels,” in 2020 IEEE 91st Vehicular Technology Conference (VTC2020-Spring), May 2020, pp. 1-6. doi: 10.1109/VTC2020-Spring48590.2020.9128408.

### DenseNet
Small convs with skips

X. Liu, D. Yang, and A. E. Gamal, “Deep neural network architectures for modulation classification,” in 2017 51st Asilomar Conference on Signals, Systems, and Computers, Oct. 2017, pp. 915-919. doi: 10.1109/ACSSC.2017.8335483.

### MCNET
Multiple Conv blocks with different kernel sizes in parallel.

T. Huynh-The, C.-H. Hua, Q.-V. Pham, and D.-S. Kim, “MCNet: An Efficient CNN Architecture for Robust Automatic Modulation Classification,” IEEE Communications Letters, vol. 24, no. 4, pp. 811-815, Apr. 2020, doi: 10.1109/LCOMM.2020.2968030.

### ResNet
5 residual convolutional blocks

T. J. O'Shea, T. Roy, and T. C. Clancy, “Over the Air Deep Learning Based Radio Signal Classification,” IEEE Journal on Selected Topics in Signal Processing, vol. 12, no. 1, pp. 168-179, Dec. 2017, doi: 10.1109/jstsp.2018.2797022.

### ResNet1
2 conv layers with a skip to more convs and MLP

X. Liu, D. Yang, and A. E. Gamal, “Deep neural network architectures for modulation classification,” in 2017 51st Asilomar Conference on Signals, Systems, and Computers, Oct. 2017, pp. 915-919. doi: 10.1109/ACSSC.2017.8335483.

### IC-AMCNET
Multi-layer conv feature extractor. Add Gaussian noise before last linear layer in MLP.

A. P. Hermawan, R. R. Ginanjar, D.-S. Kim, and J.-M. Lee, “CNN-Based Automatic Modulation Classification for Beyond 5G Communications,” IEEE Communications Letters, vol. 24, no. 5, pp. 1038-1041, May 2020, doi: 10.1109/LCOMM.2020.2970922.

### LSTM2
2-layer LSTM followed by MLP.

S. Rajendran, W. Meert, D. Giustiniano, V. Lenders, and S. Pollin, “Deep Learning Models for Wireless Signal Classification With Distributed Low-Cost Spectrum Sensors,” IEEE Transactions on Cognitive Communications and Networking, vol. 4, no. 3, pp. 433-445, Sep. 2018, doi: 10.1109/TCCN.2018.2835460.

### GRU2
2-layer GRU followed by MLP.

D. Hong, Z. Zhang, and X. Xu, “Automatic modulation classification using recurrent neural networks,” in 2017 3rd IEEE International Conference on Computer and Communications (ICCC), Dec. 2017, pp. 695--700. doi: 10.1109/CompComm.2017.8322633.

### MCLDNN
Convolutions on both IQ and I and Q seperately. Concat together and then put through 2-layer LSTM followed by MLP.

J. Xu, C. Luo, G. Parr, and Y. Luo, “A Spatiotemporal Multi-Channel Learning Framework for Automatic Modulation Recognition,” IEEE Wireless Communications Letters, vol. 9, no. 10, pp. 1629-1632, Oct. 2020, doi: 10.1109/LWC.2020.2999453.