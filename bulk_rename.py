#!/usr/bin/env python3

from typing import Generator, Tuple, Sequence, Set

import argparse
import csv
import glob
import hashlib
import itertools
import logging
import pathlib
import shutil
import string
import sys
import unicodedata

import nltk
import shtk

logging.basicConfig(encoding='utf-8', level=logging.DEBUG)
log = logging

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--format', type=str, metavar='EXT', action='append', default=['mkv','mp4','mpeg4'], help='Video Extensions to search directories for.')
    parser.add_argument('-s', '--source', type=str, metavar='PATH', action='append', default=[], help='Files or Directories containing videos to select from. Wildcards Accepted.')
    parser.add_argument('-t', '--target', type=pathlib.Path, metavar='PATH', action='append', default=[], help='Files or Directories containing subtitles with target names. ')
    parser.add_argument('-o', '--output', type=pathlib.Path, metavar='PATH', help="Output directory for renamed video copies.")
    parser.add_argument('-m', '--minsim', type=float, metavar='PERCENT', default=0.1, help='Minimum similarity for a match.')
    parser.add_argument('-d', '--dryrun', action='store_true', default=False, help='Dry run only, take no actions.')
    parser.add_argument('-q', '--quiet', action='store_true', default=False, help="Don't output info messages (Not Yet Implemented).")
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help="Output debug messages (Not Yet Implemented).")
    return parser

def load_srt(path: pathlib.Path) -> Generator[str, None, None]:
    with path.open('r', encoding="ISO-8859-1") as fin:
        state = "NUMBER"
        accum = []
        for line in fin:
            line = unicodedata.normalize('NFC', line)
            if state == "NUMBER":
                if line.strip() == "":
                    continue
                int(line.strip())
                state = "TIMESTAMP"
            elif state == "TIMESTAMP":
                state = "QUOTE"
            elif state == "QUOTE":
                if line.strip():
                    accum.append(line.strip())
                else:
                    yield " ".join(accum)
                    accum = []
                    state = "NUMBER"

        if accum:
            yield " ".join(accum)
        else:
            yield from []

def process_subs(subs: Sequence[str]) -> Set[str]:
    hashes = set()
    for sent in nltk.tokenize.sent_tokenize(" ".join(subs)):
        words = [word.lower().translate(str.maketrans('', '', string.punctuation)) for word in nltk.tokenize.word_tokenize(sent)]
        words = " ".join(word for word in words if word.strip() != "")
        hsh = hashlib.sha256()
        hsh.update(words.encode('utf-8'))
        hashes.add(hsh.hexdigest().lower())
    return hashes

def extract_subtitles(path: pathlib.Path, model='base'):
    dirpath = path.parent
    srtpath = dirpath / f"{path.stem}.whisper.{model}.srt"
    if not srtpath.exists():
        wavpath = dirpath / f"{path.stem}.wav"
        wavpath.unlink(missing_ok=True)
        log.info(f"Processing {path!s}")
        with shtk.Shell(cwd=dirpath) as sh:
            ffmpeg = sh.command('ffmpeg')
            whisper = sh.command('whisper')
            try:
                sh(ffmpeg('-i', path, '-ac', '1', wavpath.name))
            except shtk.NonzeroExitCodeException:
                return None

            sh(whisper('--model', model, '--language', 'en', wavpath.name))

        for ext in ['txt', 'vtt', 'tsv', 'json']:
            rmpath = dirpath / f"{path.stem}.{ext}"
            rmpath.unlink(missing_ok=True)

        (dirpath / f"{wavpath.name}.srt").rename(srtpath)
    else:
        log.debug(f"Skipping previously extracted subtitles for {path!s}")

    return srtpath

def main(*argv):
    args = build_parser().parse_args(argv)

    assert log is not None

    all_source_hashes = {}
    for source_file in itertools.chain.from_iterable(glob.glob(pathglob, recursive=True) for pathglob in args.source):
        source_file = pathlib.Path(source_file)
        if source_file.is_file():
            srtfile = extract_subtitles(source_file)
            try:
                if srtfile is not None:
                    all_source_hashes[source_file] = process_subs(load_srt(srtfile))
            except ValueError:
                log.exception(f"Failed to process subtitles: {source_file}")
        if 1:
            continue
        elif source.is_dir():
            for ext in args.format:
                for source_file in source.glob(f"**/*.{ext}"):
                    if source_file.is_file():
                        srtfile = extract_subtitles(source_file)
                        try:
                            if srtfile is not None:
                                all_source_hashes[source_file] = process_subs(load_srt(srtfile))
                        except ValueError:
                            log.exception(f"Failed to process subtitles: {source_file}")
        elif source.is_file():
            srtfile = extract_subtitles(source)
            try:
                if srtfile is not None:
                    all_source_hashes[source] = process_subs(load_srt(srtfile))
            except ValueError:
                log.exception(f"Failed to process subtitles: {source}")

    all_target_hashes = {}
    for target in args.target:
        if target.is_dir():
            for target_file in target.glob(f"**/*.srt"):
                log.debug(f"Loading target subtitles from {target_file!s}")
                if target_file.is_file():
                    try:
                        all_target_hashes[target_file] = process_subs(load_srt(target_file))
                    except ValueError:
                        log.exception(f"Failed to process subtitles: {target_file}")
        elif target.is_file():
            log.debug(f"Loading target subtitles from {target_file!s}")
            try:
                all_target_hashes[target] = process_subs(load_srt(target))
            except ValueError:
                log.exception(f"Failed to process subtitles: {target}")

    csvout = csv.writer(sys.stdout, delimiter=',')
    csvout.writerow(("Target", "Best Source", "Similarity"))
    for target, target_hashes in all_target_hashes.items():
        best_match, best_match_score = None, None

        for source, source_hashes in all_source_hashes.items():
            num_matches = len(source_hashes & target_hashes)
            sim = num_matches / (len(source_hashes) + len(target_hashes) - num_matches)
            if best_match is None or best_match_score < sim:
                best_match, best_match_score = source, sim

        if best_match_score > args.minsim:
            csvout.writerow((str(target), str(best_match), float(best_match_score)))
            if not args.dryrun:
                args.output.mkdir(exist_ok=True)
                outpath = args.output / f"{target.stem}{best_match.suffix}"
                logging.info(f"Copying {best_match!s} -> {outpath!s}")
                shutil.copy2(best_match, outpath, follow_symlinks=True)

if __name__ == "__main__":
    main(*sys.argv[1:])
