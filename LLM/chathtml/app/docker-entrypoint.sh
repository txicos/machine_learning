
set -e

whoami
echo $PWD
echo $PATH

if [ -n "$1" ]; then
    exec "$@"
fi

