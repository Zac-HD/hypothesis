RELEASE_TYPE: patch

This patch makes writing :ref:`observability <observability>` data to
``.hypothesis/observed`` cheaper under multiple threads.  Instead of holding a
single global lock for every write, each file now has its own queue; an
enqueuing thread writes whatever has been enqueued if it can acquire the lock,
and otherwise leaves the work for whichever thread currently holds it.
