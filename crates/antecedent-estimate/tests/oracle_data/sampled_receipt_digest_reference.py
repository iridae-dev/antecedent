"""Independent derivation of the sampled-recovery receipt digest golden vector.

Framing (from the documented canonical writer): little-endian u64 words, length-prefixed
UTF-8 text and byte strings, reals as IEEE-754 little-endian bit patterns; digest = BLAKE3 of
the stream (implemented here from scratch for inputs of at most one chunk).

    python3 sampled_receipt_digest_reference.py

Prints the vector for the former calibration text `unmeasured` (9703333f...) and for the
current `not_measured` (f431887c...).

SPDX-License-Identifier: MIT OR Apache-2.0
"""
import struct
M=0xFFFFFFFF
IV=[0x6A09E667,0xBB67AE85,0x3C6EF372,0xA54FF53A,0x510E527F,0x9B05688C,0x1F83D9AB,0x5BE0CD19]
PERM=[2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8]
CS,CE,ROOT=1,2,8
def rotr(x,n): return ((x>>n)|(x<<(32-n)))&M
def g(s,a,b,c,d,mx,my):
    s[a]=(s[a]+s[b]+mx)&M; s[d]=rotr(s[d]^s[a],16); s[c]=(s[c]+s[d])&M; s[b]=rotr(s[b]^s[c],12)
    s[a]=(s[a]+s[b]+my)&M; s[d]=rotr(s[d]^s[a],8); s[c]=(s[c]+s[d])&M; s[b]=rotr(s[b]^s[c],7)
def compress(cv,block,counter,blen,flags):
    m=list(struct.unpack('<16I',block.ljust(64,b'\0')))
    s=cv[:]+IV[:4]+[counter&M,(counter>>32)&M,blen,flags]
    for r in range(7):
        g(s,0,4,8,12,m[0],m[1]);g(s,1,5,9,13,m[2],m[3]);g(s,2,6,10,14,m[4],m[5]);g(s,3,7,11,15,m[6],m[7])
        g(s,0,5,10,15,m[8],m[9]);g(s,1,6,11,12,m[10],m[11]);g(s,2,7,8,13,m[12],m[13]);g(s,3,4,9,14,m[14],m[15])
        if r<6: m=[m[i] for i in PERM]
    return [s[i]^s[i+8] for i in range(8)]+[s[i+8]^cv[i] for i in range(8)]
def blake3(data):
    assert len(data)<=1024
    blocks=[data[i:i+64] for i in range(0,len(data),64)] or [b'']
    cv=IV[:]
    for i,b in enumerate(blocks):
        f=(CS if i==0 else 0)|((CE|ROOT) if i==len(blocks)-1 else 0)
        out=compress(cv,b,0,len(b),f)
        cv=out[:8]
    return struct.pack('<8I',*cv).hex()
class W:
    def __init__(s,tag): s.b=b''; s.text(tag)
    def word(s,x): s.b+=struct.pack('<Q',x)
    def bytes(s,d): s.word(len(d)); s.b+=d
    def text(s,t): s.bytes(t.encode())
    def real(s,x): s.b+=struct.pack('<d',x)
def digest(calibration):
    w=W("sampled_recovery.receipt.bca.v1")
    w.text("snap"); w.word(4); w.text("deriv"); w.text("in")
    w.word(2000); w.word(99); w.real(0.0); w.real(0.25); w.real(5.0); w.real(1.0); w.real(0.0)
    w.real(0.4); w.real(0.95); w.real(0.1); w.real(0.7); w.text(calibration)
    w.word(2)
    w.word(0); w.text("sel0"); w.bytes(b'\1'); w.real(0.4); w.bytes(b'\0'); w.word(0)
    w.word(1); w.text("sel1"); w.bytes(b'\0'); w.bytes(b'\1'); w.text("zero cell"); w.word(1); w.bytes(bytes([1,0,1]))
    w.text("bootstrap_bca"); w.text("midrank_exact_ties:type7:delete_one_row")
    w.bytes(b'\1'); w.real(0.1); w.real(-0.2); w.real(0.025); w.real(0.975)
    w.word(1); w.bytes(bytes([3,1,2])); w.word(4); w.real(0.35)
    return blake3(w.b)
if __name__=="__main__":
    print(digest("unmeasured")); print(digest("not_measured"))
