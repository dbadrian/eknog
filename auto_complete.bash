#!/bin/bash

# Requires Python Module: argcompd

_complete_example()
{
    local completions=$("${1}" --_complete "${COMP_CWORD}" "${COMP_WORDS[@]}")
    if [ $? -eq 0 ]; then
        mapfile -t COMPREPLY < <(echo -n "${completions}")
    fi
}

complete -F _complete_example eknog.py
