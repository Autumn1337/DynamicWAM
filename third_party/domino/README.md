# DOMINO source reconstruction

The evaluated DOMINO revision
`9b1f34129323957baff35bcbc866fceca7e02779` is a local descendant of the
public upstream revision
`93d485a34a6e710bb4c895d83e2bb0520a9f169e`. Because the evaluated revision is
not available from the public upstream remote, this directory retains its
complete binary-safe source delta as `evaluated.patch`.

`scripts/bootstrap_domino.py` fetches the public base, applies the patch under
`external/DOMINO`, and accepts the result only when both identities match:

- reconstructed tree SHA-256:
  `54dd0280e44ea0539274365b7feaeba24e36b7629e8679c2999476addd9c314c`;
- `script/eval_policy.py` SHA-256:
  `fec0002fe1e30833b225d3559b09b583a9a4f7a22c29bc54a3817eb70d1aea38`.

The patch SHA-256 is
`52e8b8dd18523b62daab14efd0f1acd60a5cf93fef257ab1d21294fa43f506ee`.
The upstream Apache-2.0 license is retained alongside it.
