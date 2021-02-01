# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0


class _BaseMitigation:
    """
    Base class to hold properties of Binary Mitigation Techniques.

    This module is not intended to be used directly, instead it declares
    the basic skeleton of binary mitigation class together with a set of parameters
    that are common to every mitigation technique.

    It declares properties that sub-classes will initiate according to each fairness metric.
    Note that the properties ought to not be changed by the end user and are private accordingly.

        - ``name`` property to display name of the class
        - ``description`` property to display description of the class

    Attributes
    ----------
    _name: str
        The name of the bias mitigation technique.
    _description: str
        The description of the bias mitigation technique

    """

    def __init__(self, name, description):
        self._name = name
        self._description = description

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description
