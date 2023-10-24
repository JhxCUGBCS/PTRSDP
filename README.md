# PTRSDP
Military relation extraction based on Prompt-tuning and dependency syntactic analysis.<br>
We provide a new Chinese military relation extraction dataset.

# Dataset Source
The compilation of the military dataset involved the utilization of diverse sources, including the Global Military Network, Wikidata military equipment data, Baidu Encyclopedia, and corpus data from third-party websites. These sources provided unstructured or semi-structured data, which were collected through the employment of web crawler technology. Following the preprocessing of the acquired data, which entailed tasks such as long text segmentation and logical error correction, the construction of the military relation extraction dataset was accomplished through manual annotation. During the annotation process, the text was classified into five relation types with reference to this research: instance, operation, country of occurrence, participation in combat, and equipment. Subsequently, the relation type, head entity, and tail entity were sequentially labeled. Finally, the resulting dataset consisted of 12,607 labeled instances, which were subsequently partitioned into training, testing, and validation sets in an 8:1:1 ratio.
![Data Crawling Flowchart.](https://github.com/JhxCUGBCS/PTRSDP/blob/main/%E6%95%B0%E6%8D%AE%E7%88%AC%E5%8F%96%E6%B5%81%E7%A8%8B.jpg)
