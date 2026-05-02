open_checkpoint /home/user/red_eyes_is_all_you_need/build/post_route.dcp

# For each target hierarchy, report bounding box and resource counts of currently placed cells

set targets {
  pb_qkv        u_tf/u_tl/u_attn/u_qkv
  pb_proj       u_tf/u_tl/u_attn/u_proj
  pb_ff_up      u_tf/u_tl/u_ff_up
  pb_ff_down    u_tf/u_tl/u_ff_down
  pb_head_proj  u_tf/u_head_proj
  pb_ln_f       u_tf/u_ln_f
  pb_softmax    u_tf/u_tl/u_attn/u_sm
  pb_attn       u_tf/u_tl/u_attn
  pb_tl         u_tf/u_tl
}

proc bbox_for {hier} {
  set cells [get_cells -quiet -hier -filter "PARENT =~ $hier* && IS_PRIMITIVE && LOC != \"\""]
  if {[llength $cells] == 0} { return "" }

  array set min_x_t {}
  array set max_x_t {}
  array set min_y_t {}
  array set max_y_t {}
  array set count_t {}

  foreach c $cells {
    set site [get_property LOC $c]
    set ref  [get_property REF_NAME $c]
    set ttype "OTHER"
    switch -regexp -- $ref {
      "^DSP48"    { set ttype "DSP" }
      "^RAMB36"   { set ttype "RAMB36" }
      "^RAMB18"   { set ttype "RAMB18" }
      "^FD"       { set ttype "FF" }
      "^LUT"      { set ttype "LUT" }
      "^RAMD"     { set ttype "RAMD" }
      "^RAMB"     { set ttype "RAMB" }
    }
    set sname [get_property NAME $site]
    if {[regexp {_X(\d+)Y(\d+)} $sname -> x y]} {
      if {![info exists min_x_t($ttype)]} {
        set min_x_t($ttype) $x; set max_x_t($ttype) $x
        set min_y_t($ttype) $y; set max_y_t($ttype) $y
        set count_t($ttype) 0
      } else {
        if {$x < $min_x_t($ttype)} {set min_x_t($ttype) $x}
        if {$x > $max_x_t($ttype)} {set max_x_t($ttype) $x}
        if {$y < $min_y_t($ttype)} {set min_y_t($ttype) $y}
        if {$y > $max_y_t($ttype)} {set max_y_t($ttype) $y}
      }
      incr count_t($ttype)
    }
  }

  set lines {}
  foreach t [lsort [array names count_t]] {
    lappend lines [format "    %-7s count=%-5d X=%d..%d Y=%d..%d" \
      $t $count_t($t) $min_x_t($t) $max_x_t($t) $min_y_t($t) $max_y_t($t)]
  }
  return [join $lines "\n"]
}

puts "==================================================================="
puts "PLACEMENT BOUNDING BOXES (current post-route)"
puts "==================================================================="
foreach {pbname hier} $targets {
  puts ""
  puts "=== $pbname  ($hier) ==="
  set b [bbox_for $hier]
  if {$b == ""} {
    puts "  (no cells found)"
  } else {
    puts $b
  }
}

puts ""
puts "==================================================================="
puts "DEVICE GRID (resource columns and ranges)"
puts "==================================================================="
foreach rt {DSP48E1 RAMB36E1 RAMB18E1} {
  set sites [get_sites -filter "SITE_TYPE == $rt"]
  array set col_y_min {}
  array set col_y_max {}
  array set col_count {}
  foreach s $sites {
    set sname [get_property NAME $s]
    if {[regexp {_X(\d+)Y(\d+)} $sname -> x y]} {
      if {![info exists col_count($x)]} {
        set col_y_min($x) $y; set col_y_max($x) $y; set col_count($x) 0
      } else {
        if {$y < $col_y_min($x)} {set col_y_min($x) $y}
        if {$y > $col_y_max($x)} {set col_y_max($x) $y}
      }
      incr col_count($x)
    }
  }
  puts ""
  puts "  $rt columns:"
  foreach x [lsort -integer [array names col_count]] {
    puts [format "    X=%d  count=%-3d  Y=%d..%d" $x $col_count($x) $col_y_min($x) $col_y_max($x)]
  }
  array unset col_y_min; array unset col_y_max; array unset col_count
}
