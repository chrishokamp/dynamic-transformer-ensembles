import os
import json
import pickle
import datetime
import codecs
import gzip
import io
import csv
import random
import shutil


random.seed(24)
csv.field_size_limit(1000000)


def force_mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def wipe_file(path):
    with open(path, 'w') as f:
        pass


def abs_listdir(dir):
    paths = [os.path.join(dir, x) for x in os.listdir(dir)]
    return sorted(paths)


def readfile(path):
    with open(path) as f:
        text = f.read()
    return text


def readlines(path):
    with open(path) as f:
        text = f.read()
    return text.split('\n')


def writefile(s, path):
    with open(path, 'w') as f:
        f.write(s)


def writelines(lines, path):
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def readjson(path):
    text = readfile(path)
    return json.loads(text)


def writejson(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f)


def load_pkl(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def dump_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj,  f)


def periodic_print(i, n=10):
    if i % n == 0:
        print(i)


def append_to_file(s, path):
    with open(path, 'a') as f:
        f.write(s)


def split_dataset(inpath, trainpath, testpath, ratio=0.5):
    lines = readlines(inpath)
    split_at = int(ratio * len(lines))
    trainlines = lines[:split_at]
    testlines = lines[split_at:]
    writelines(trainlines, trainpath)
    writelines(testlines, testpath)


def get_best_scored(d):
    return sorted(d, key=lambda x: x[1], reverse=True)[0][0]


def parse_isodate(s):
    return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ")


def readfile2(path):
    with codecs.open(path, "r", encoding='utf-8', errors='ignore') as f:
        text = f.read()
    return text


def write_gzip(text, path):
    with gzip.open(path, 'wb') as output:
        with io.TextIOWrapper(output, encoding='utf-8') as enc:
            enc.write(text)


def read_gzip(path):
    with gzip.open(path, 'rb') as input_file:
        with io.TextIOWrapper(input_file) as dec:
            content = dec.read()
    return content


def select_all_nth(list_of_lists, n):
    return [x[n] for x in list_of_lists]


def get_date_range(start, end):
    diff = end - start
    date_range = []
    for n in range(diff.days + 1):
        t = start + datetime.timedelta(days=n)
        date_range.append(t)
    return date_range


def read_sheet(path, delimiter=','):
    '''Read csv, tsv, etc. file'''
    with open(path, 'r') as f:
        row_dicts = []
        rows = list(csv.reader(f, delimiter=delimiter))
        header = rows[0]
        idx_map = dict((x, i) for i, x in enumerate(header))
        for row in rows[1:]:
            row_dict = {}
            for x, i in idx_map.items():
                if i < len(row):
                    row_dict[x] = row[i]
            row_dicts.append(row_dict)
    return row_dicts


def write_sheet(sheet, path, header=None, write_header=True):
    if header is None:
        header = sorted(sheet[0].keys())
    with open(path, 'w') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        for item in sheet:
            row = []
            for k in header:
                if k in item:
                    v = item[k]
                else:
                    v = ''
                row.append(v)
            writer.writerow(row)


def read_jsonl(path, load=False, start=0, stop=None):

    def read_jsonl_stream(path):
        with open(path) as f:
            for i, line in enumerate(f):
                if (stop is not None) and (i >= stop):
                    break
                if i >= start:
                    yield json.loads(line)

    data = read_jsonl_stream(path)
    if load:
        data = list(data)
    return data


def write_jsonl(items, path, batch_size=100, override=True):
    if override:
        with open(path, 'w'):
            pass

    batch = []
    for i, x in enumerate(items):
        if i > 0 and i % batch_size == 0:
            with open(path, 'a') as f:
                output = '\n'.join(batch) + '\n'
                f.write(output)
            batch = []
        raw = json.dumps(x)
        batch.append(raw)

    if batch:
        with open(path, 'a') as f:
            output = '\n'.join(batch) + '\n'
            f.write(output)


def read_tap_dataset_csv(path):
    data = []
    with open(path) as f:
        reader = csv.reader(f)
        for text, label in reader:
            item = {
                'text': text,
                'label': label
            }
            data.append(item)
    return data


def shuffled(items):
    items = items.copy()
    random.shuffle(items)
    return items


def sample(items, n):
    return shuffled(items)[:n]
