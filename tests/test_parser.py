from utils import Parser


def test_parser():
    """ test that parser returns the correct types
    """
    sys_args = ["--data",  "observations.csv", "--outfile", "results/samples.dat",
                "--summarise", "--basis", "cycle", "--treeprior", "all", "-r", "6",
                "-p", ".75", "--cob-freq", "100", "--seed", "123"]

    parser = Parser.Parser(sys_args)
    assert(isinstance(parser.args.data, str))
    assert(isinstance(parser.args.outfile, str))
    assert(isinstance(parser.args.summarise, bool))
    assert(isinstance(parser.args.basis, str))
    assert(isinstance(parser.args.treeprior, str))
    assert(isinstance(parser.args.r, int))
    assert(isinstance(parser.args.p, float))
    assert(isinstance(parser.args.cob_freq, int))
    if parser.args.seed is not None:
        assert(isinstance(parser.args.seed, int))