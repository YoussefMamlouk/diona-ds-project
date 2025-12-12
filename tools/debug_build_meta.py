import tempfile, os, shutil
import setuptools.build_meta as bm

md = tempfile.mkdtemp(prefix='debug_meta_')
print('METADATA_DIR:', md)

# monkeypatch to print walk
import inspect

print('build_meta members:', [n for n in dir(bm) if not n.startswith('__')])

Backend = getattr(bm, '_BuildMetaBackend', None)
print('Has _BuildMetaBackend:', Backend is not None)
if Backend:
    print('Backend attrs:', [n for n in dir(Backend) if not n.startswith('__')])

try:
    res = bm.prepare_metadata_for_build_editable(md)
    print('RESULT:', res)
except Exception as e:
    print('EXCEPTION:', type(e), e)
finally:
    # list temp dir contents
    print('METADATA_DIR_CONTENTS:')
    for root, dirs, files in os.walk(md):
        print('-', root)
        for d in dirs:
            print('   D:', d)
        for f in files:
            print('   F:', f)
    shutil.rmtree(md, ignore_errors=True)
