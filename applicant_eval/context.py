"""Conservative English claim-context flags; never infer ability from missing text.

These flags guard a general-purpose NLI model, which can otherwise attribute a
colleague's work to the applicant. Ambiguous text remains available to reviewers.
This is deliberately not an exhaustive language understanding system.
"""

import re

PATTERNS = {
    "instruction_or_requirement": r"ignore (?:all |any )?(?:previous|prior) instructions|mark (?:this |the )?applicant|output (?:the |a )?(?:highest|score)|job requirements|requirements for|topics of interest",
    "prospective_or_hypothetical": r"\b(?:plan(?:ning)?|intend|hope|want|wish|propos(?:e|ed|al))\b|\bwould\b|\bif I had\b|\bnot (?:yet )?started\b|\bwill (?:build|write|implement|learn)\b",
    "other_people_or_ambiguous_ownership": r"\b(?:recruit(?:ed|ing)?|hired?|hiring)\b|\b(?:engineers|developers|contractors|others|colleagues)\b.{0,90}\b(?:built|wrote|write|performed|implemented|maintained)\b|\b(?:my|our|the) (?:team|organization)\b|\b(?:sold|marketed)\b|\b(?:managed|coordinated) (?:the )?(?:timelines|team|project)\b",
    "explicit_negation": r"\bnever\b|\bnot\b(?!\s+only)|\b\w+n['’]t\b|\bnone\b|\bno (?:actual |relevant |work )?experience\b|\bnobody\b|\black(?:s|ing)?\b.{0,30}\bexperience\b",
}


def context_flags(quote):
    return [name for name, pattern in PATTERNS.items() if re.search(pattern, quote, re.I)]
