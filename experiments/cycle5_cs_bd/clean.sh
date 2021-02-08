#!/bin/bash
DIR=$(dirname $0) 
[ -e "$DIR/Data.pkl" ] && rm "$DIR/Data.pkl"
[ -e "$DIR/Params.pkl" ] && rm "$DIR/Params.pkl"
[ -e "$DIR/K_matrices.pkl" ] && rm "$DIR/K_matrices.pkl"
[ -e "$DIR/log.dat" ] && rm "$DIR/log.dat"

