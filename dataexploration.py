import random
import numpy as np
import pandas as pd
import GEOparse
import multiprocessing as mp

uIDToSymbol = None
blood_samples = None


def ProcessSample(sample_):
    sample_name, sample, id, gene, condition = sample_

    def IDToSymbol(id):
        try:
            return gene[gene['ID'] == id]['Gene Symbol'].iloc[0]
        except:
            return "NOSYMBOLFOUND"

    uIDToSymbol = np.frompyfunc(IDToSymbol, 1, 1)

    sample['ID_REF'] = uIDToSymbol(sample['ID_REF'].to_numpy())
    sample = sample[sample['ID_REF'] != 'NOSYMBOLFOUND'].transpose()
    columns = sample.iloc[0]
    sample = sample[1:]
    sample.columns = columns
    sample.insert(0, 'Name', sample_name)
    sample.insert(1, 'Condition', condition)
    sample = sample.loc[:, ~sample.columns.duplicated()]
    print(f"Processed sample {id}.")
    return sample


def ReadDataset():
    global uIDToSymbol
    global blood_samples
    #-------- Dataset Administration -----------
    gse = GEOparse.get_GEO(geo="GSE45291",
                           destdir="./Dataset",
                           silent=False,
                           include_data=True)
    #-------------------------------------------

    gene = gse.gpls['GPL13158'].table.dropna(axis=0,
                                             subset=['Gene Symbol'
                                                     ])[['ID', 'Gene Symbol']]
    columns = ['Name', 'Condition'] + list(
        set(gene['Gene Symbol'].to_numpy().tolist()))
    blood_samples = pd.DataFrame(columns=columns)

    RA = []
    SLE = []
    Control = []

    for sample_id in gse.gsms:
        disease = gse.gsms[sample_id].metadata['characteristics_ch1'][2]
        if disease == 'disease: Control':
            Control.append(sample_id)
        elif 'Rheumatoid Arthiritis' in disease:
            RA.append(sample_id)
        elif 'SLE' in disease:
            SLE.append(sample_id)
        else:
            print(disease)

    RA = random.sample(RA, 20)
    SLE = random.sample(SLE, 20)

    samples = []
    for i, sample_id in enumerate([*RA, *SLE, *Control]):
        print(f"Reading sample {i}")
        condition = gse.gsms[sample_id].metadata['characteristics_ch1'][2]
        samples.append(
            (sample_id, gse.gsms[sample_id].table.copy(deep=True), i,
             gene.copy(deep=True), condition))

    return samples


if __name__ == '__main__':
    samples = ReadDataset()
    print(f"Starting Sample Processing")
    p = mp.Pool(mp.cpu_count())
    res = p.map(ProcessSample, samples)

    for sample in res:
        blood_samples = blood_samples.append(sample, ignore_index=True)

    blood_samples.head()

    blood_samples.to_csv('./SLE_RA_Control_blood_panels.csv')
