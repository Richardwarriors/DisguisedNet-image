import splitfolders

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio("CVC\Encrpyted\AES\AESB4N4", output="CVC\Encrpyted\AES\AESB4N4-1",
    seed=1337, ratio=(.7, .15, .15), group_prefix=None, move=False) # default values
