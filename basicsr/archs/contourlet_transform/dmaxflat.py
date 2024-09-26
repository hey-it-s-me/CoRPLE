# -*- coding: utf-8 -*-
#    PyContourlet
#
#    A Python library for the Contourlet Transform.
#
#    Copyright (C) 2011 Mazay Jiménez
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation version 2.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

from numpy import *


def dmaxflat(N, d):
    """returns 2-D diamond maxflat filters of order 'N'
    the filters are nonseparable and 'd' is the (0,0) coefficient,
    being 1 or 0 depending on use
    """
    if (N > 7 or N < 1):
        raise ValueError("N must be in {1,2,3,4,5,6,7}")

    if equal(N, 1):
        h = array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4.0
        h[2, 2] = d
        return h
    elif equal(N, 2):
        h = array([[0, -1, 0], [-1, 0, 10], [0, 10, 0]])
        h = hstack((h, fliplr(h[:, :-1])))
        h = vstack((h, flipud(h[:-1, :]))) / 32.0
        h[3, 3] = d
        return h
    elif equal(N, 3):
        h = array([[0, 3, 0, 2],
                   [3, 0, -27, 0],
                   [0, -27, 0, 174],
                   [2, 0, 174, 0]])
        h = hstack((h, fliplr(h[:, :-1])))
        h = vstack((h, flipud(h[:-1, :]))) / 512.0
        h[4, 4] = d
        return h
    elif equal(N, 4):
        h = array([[0, -5, 0, -3, 0],
                   [-5, 0, 52, 0, 34],
                   [0, 52, 0, -276, 0],
                   [-3, 0, -276, 0, 1454],
                   [0, 34, 0, 1454, 0]]) / 2.0**12
        h = hstack((h, fliplr(h[:, :-1])))
        h = vstack((h, flipud(h[:-1, :])))
        h[5, 5] = d
        return h
    elif equal(N, 5):
        h = array([[0, 35, 0, 20, 0, 18],
                   [35, 0, -425, 0, -250, 0],
                   [0, -425, 0, 2500, 0, 1610],
                   [20, 0, 2500, 0, -10200, 0],
                   [0, -250, 0, -10200, 0, 47780],
                   [18, 0, 1610, 0, 47780, 0]]) / 2.0**17
        h = hstack((h, fliplr(h[:, :-1])))
        h = vstack((h, flipud(h[:-1, :])))
        h[6, 6] = d
        return h
    elif equal(N, 6):
        h = array([[0, -63, 0, -35, 0, -30, 0],
                   [-63, 0, 882, 0, 495, 0, 444],
                   [0, 882, 0, -5910, 0, -3420, 0],
                   [-35, 0, -5910, 0, 25875, 0, 16460],
                   [0, 495, 0, 25875, 0, -89730, 0],
                   [-30, 0, -3420, 0, -89730, 0, 389112],
                   [0, 44, 0, 16460, 0, 389112, 0]]) / 2.0**20
        h = hstack((h, fliplr(h[:, :-1])))
        h = vstack((h, flipud(h[:-1, :])))
        h[7, 7] = d
        return h
    elif equal(N, 7):
        h = array([[0, 231, 0, 126, 0, 105, 0, 100],
                   [231, 0, -3675, 0, -2009, 0, -1715, 0],
                   [0, -3675, 0, 27930, 0, 15435, 0, 13804],
                   [126, 0, 27930, 0, -136514, 0, -77910, 0],
                   [0, -2009, 0, -136514, 0, 495145, 0, 311780],
                   [105, 0, 15435, 0, 495145, 0, -1535709, 0],
                   [0, -1715, 0, -77910, 0, -1535709, 0, 6305740],
                   [100, 0, 13804, 0, 311780, 0, 6305740, 0]]) / 2.0**24
        h = hstack((h, fliplr(h[:, :-1])))
        h = vstack((h, flipud(h[:-1, :])))
        h[8, 8] = d
        return h
