"""
This module provides utility functions for list and string operations, as well as
a function to generate seeds based on a member number for probtest.
"""

import re


def unique_elements(inlist):
    unique = []
    for element in inlist:
        if element not in unique:
            unique.append(element)
    return unique

def prepend_type_to_member_id(member_type, member_id):
    return (member_type + "_" + str(member_id)) if member_type else str(member_id)


def first_idx_of(li, el):
    return li.index(el)


def last_idx_of(li, el):
    return len(li) - li[::-1].index(el) - 1


def numbers(s):
    "join all numbers from a string and return as an int (base 10)"
    return int("".join(re.findall(r"\d+", s)))


def get_seed_from_member_id(member_id: int) -> int:
    """
    Returns the seed corresponding to the given member number.
    """
    if member_id <= 0 or member_id > 120:
        raise ValueError(
            "Invalid member number: only values between 1 and 120 are valid."
        )

    seeds = [
        2547566401,
        3390797009,
        2583677123,
        2473118423,
        2143626467,
        4173353531,
        1628145667,
        3371547601,
        3301010651,
        2632851427,
        2655986491,
        439228313,
        2037313633,
        2681973191,
        92727749,
        424192897,
        3575206447,
        1431770341,
        2158581499,
        1389519269,
        3860054131,
        1423646869,
        100708747,
        271345577,
        2226813817,
        541149953,
        4244349953,
        3656774221,
        319908709,
        2563334467,
        2130688607,
        240773417,
        2530154009,
        3415037839,
        1546988207,
        2804242241,
        2454029327,
        3076953343,
        2214797017,
        148621783,
        919008731,
        3073167649,
        2544996131,
        1560372031,
        747252683,
        676033667,
        1621633327,
        1497119137,
        2115885181,
        3052669511,
        363171217,
        2241551287,
        1013429699,
        2261903797,
        2731086991,
        2705856533,
        2958671119,
        1063346831,
        2497761787,
        3199470749,
        2719907293,
        1367554031,
        431355971,
        1508132207,
        2657310209,
        515649373,
        422364193,
        1538703743,
        4215084967,
        3862569617,
        3112572893,
        2766389459,
        1220880631,
        1925252027,
        1314078553,
        2728133203,
        3409408591,
        1924140919,
        3036144263,
        4211976127,
        2710652809,
        4294706461,
        2258428867,
        3141914599,
        2581251989,
        395560769,
        3630067789,
        2785597153,
        1599809011,
        3586998517,
        3974001257,
        1972680337,
        385683763,
        1766890739,
        3719579741,
        74170891,
        165779983,
        2615058419,
        412023839,
        3993305273,
        1678330057,
        1202490409,
        3263858743,
        2986703311,
        3476257331,
        2694509239,
        3275559121,
        3515429279,
        2871479801,
        490303711,
        1979395757,
        1959584257,
        819309853,
        2907252791,
        4029093293,
        2289808813,
        2119985837,
        121836503,
        3955242553,
        736607237,
    ]

    seed = seeds[member_id - 1]

    return seed
