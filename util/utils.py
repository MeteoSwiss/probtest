import hashlib


def unique_elements(inlist):
    unique = []
    for element in inlist:
        if element not in unique:
            unique.append(element)
    return unique


def first_idx_of(list, element):
    return list.index(element)


def last_idx_of(list, element):
    return len(list) - list[::-1].index(element) - 1


def generate_seed_from_member_id(member_id, use_64_bits=True):
    if use_64_bits:
        return int.from_bytes(
            hashlib.sha256(member_id.encode("utf-8")).digest()[:8],
            byteorder="little",
            signed=True,
        )
    else:
        return int.from_bytes(
            hashlib.sha256(member_id.encode("utf-8")).digest()[:4],
            byteorder="little",
            signed=True,
        )
