import re
import sys
from datetime import datetime

import numpy as np

from util.constants import datetime_format
from util.log_handler import logger

timing_start_regex = r"(?: +L? ?[a-zA-Z_.]+)"
timing_element_regex = r"(?:\[?\d+[.msh]?\d*s?\]? +)"
timing_regex = timing_start_regex + " +" + timing_element_regex + "{11} *(?!.)"
header_regex = (
    r" name *# calls *t_min *min rank *t_avg *"
    + r"t_max *max rank *total min \(s\) *total min rank *"
    + r"total max \(s\) *total max rank * total avg \(s\)"
)
indent_regex = r"^ *L? "
hour_regex = r"(\d+)h(\d+)m(\d+)s"
minute_regex = r"(\d+[.]?\d*)m(\d+[.]?\d*)s"
sec_regex = r"(\d+[.]?\d*)s"
number_regex = r"(\d+[.]?\d*)"

dateline_regexs = (
    r"(?:[A-Z][a-z]{2} +){2}\d{1,2} \d{2}:\d{2}:\d{2} [A-Z]{3,4} 20\d{2}",
    (
        r"(?:[A-Z][a-z]{2} +)\d{1,2} (?:[A-Z][a-z]{2} +)20\d{2} \d{2}:\d{2}:\d{2} "
        "[A-Z]{2} [A-Z]{3,4}"
    ),
)
icon_date_formats = ("%a %b %d %H:%M:%S %Z %Y", "%a %d %b %Y %H:%M:%S %p %Z")

dict_regex = "({} *:) *(.*)"


def _convert_dateline_to_start_end_datetime(dateline, icon_date_format):
    # LOG.check files have more dates than we need
    # The dates we are interested in are always at the same position relative to the
    #  other dates
    if len(dateline) > 2:
        dateline = [dateline[1], dateline[2]]

    start_time, finish_time = dateline

    finish_datetime = datetime.strptime(finish_time, icon_date_format)
    finish_datetime_converted = finish_datetime.strftime(datetime_format)

    start_datetime = datetime.strptime(start_time, icon_date_format)
    start_datetime_converted = start_datetime.strftime(datetime_format)

    return (start_datetime_converted, finish_datetime_converted)


def read_logfile(filename):
    with open(filename, "r", encoding="latin-1") as f:
        # read file into list of lines, remove empty lines
        full_file = f.read()
        data = [e for e in full_file.split("\n") if e != ""]

        # filter by timing headers and elements
        data = [
            e for e in data if re.search(header_regex, e) or re.search(timing_regex, e)
        ]

        # store line numbers of timing table headers
        header_lines = [i for i, e in enumerate(data) if re.search(header_regex, e)]

        # initialize storage for all tables
        timing_data = []

        # construct timing tables
        for k, i_header in enumerate(header_lines):
            # make sure stay within header_line bounds
            i_end = header_lines[k + 1] if k + 1 < len(header_lines) else -1

            # get data from this table (starting one line after header)
            table = data[i_header + 1 : i_end]

            # parse table header
            header_elements = [
                e.lstrip().rstrip()
                for e in data[i_header].split("  ")
                if e not in ["", " "]
            ]
            timing_data_k = {e: [] for e in header_elements}

            # parse table elements
            timing_data_k["indent"] = []
            timing_data_k["name"] = []

            for table_line in table:
                elements = [
                    e.replace("[", "").replace("]", "")
                    for e in table_line.split(" ")
                    if e not in ["", "L"]
                ]
                if len(elements) != len(header_elements):
                    logger.critical(
                        (
                            "Number of header elements ({}) "
                            + "does not match number of table elements ({})"
                        ).format(len(header_elements), len(elements))
                    )
                    logger.critical("header: {}".format(" -- ".join(header_elements)))
                    logger.critical("table : {}".format(" -- ".join(elements)))
                    sys.exit(1)

                # find indentation level for each table line
                first = re.search(indent_regex, table_line).group(0)
                # assume 1 indent is 3 white spaces
                timing_data_k["indent"].append(len(first) // 3)

                timing_data_k["name"].append(elements[0])
                for i in np.arange(1, len(elements)):
                    timing_data_k[header_elements[i]].append(parse_time(elements[i]))

            timing_data.append(timing_data_k)

        # start parsing meta data from log
        meta_data = {}

        # get start and finish time from job
        found_dateline_yes = False
        for dateline_regex, icon_date_format in zip(dateline_regexs, icon_date_formats):
            dateline = re.findall(dateline_regex, full_file)

            if dateline:
                (
                    start_datetime_converted,
                    finish_datetime_converted,
                ) = _convert_dateline_to_start_end_datetime(dateline, icon_date_format)
                found_dateline_yes = True

        if not found_dateline_yes:
            raise Exception("Could not match any regex for start and end time.")

        meta_data["start_time"] = start_datetime_converted
        meta_data["finish_time"] = finish_datetime_converted

        # get meta data from ICON log (in the form "Key : Value")
        revision = re.search(dict_regex.format("Revision"), full_file)
        branch = re.search(dict_regex.format("Branch"), full_file)

        meta_data["revision"] = revision.group(2)
        meta_data["branch"] = branch.group(2)

    meta_data["n_tables"] = len(timing_data)
    meta_data["entries"] = [len(e["indent"]) for e in timing_data]

    return timing_data, meta_data


def parse_time(time_string):
    m1 = re.match(hour_regex, time_string)
    m2 = re.match(minute_regex, time_string)
    m3 = re.match(sec_regex, time_string)
    m4 = re.match(number_regex, time_string)
    if m1:
        h, m, s = [m1.group(i) for i in [1, 2, 3]]
    elif m2:
        m, s = [m2.group(i) for i in [1, 2]]
        h = 0
    elif m3:
        s = m3.group(1)
        h = 0
        m = 0
    elif m4:
        s = m4.group(0)
        h = 0
        m = 0
    else:
        logger.error("did not match regex")
    out = float(h) * 60 * 60 + float(m) * 60 + float(s)
    return out
