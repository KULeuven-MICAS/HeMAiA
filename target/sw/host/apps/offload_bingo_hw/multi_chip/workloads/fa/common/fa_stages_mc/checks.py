# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""FaStages' checks (the checks themselves are FaBuild's) and finish(): the checks wait for every owner's puts;
keys_by_chunk orders the collectives and checks the key transfers against the reduce chains."""


class FaStagesChecks:

    def run_checks(self, checks):
        """The key checks read what every owner's puts wrote into its L3: the checks also wait for every put."""
        puts = list(getattr(self, "kv_put", {}).values()) + list(getattr(self, "kvp_put", {}).values())
        self.check_copies = list(self.check_copies) + [t for t in puts if t not in self.check_copies]
        return super().run_checks(checks)

    def finish(self, app_name, **kw):
        if self.keys_by_chunk:
            self._order_collectives()
            self._check_kx_after_chains()
        super().finish(app_name, **kw)
