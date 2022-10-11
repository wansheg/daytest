
namespace eval ::AP::JSON_UTILS {

variable HUDDLE_IMPL_INIT 0

proc json_init {} {
    variable HUDDLE_IMPL_INIT
    if { $HUDDLE_IMPL_INIT } {
        #puts "DBG: skipping json_init"
        return
    }
    set parent_ns "::AP::JSON_UTILS::"
    set HUDDLE_IMPL_INIT 1
    package require huddle
    package require huddle::json

    #puts "DBG: running json_init with huddle [package present huddle]"
    set reflect_huddle_cmds [list keys get gets equal type llength remove combine strip]

    if { [package vcompare [package present huddle] 0.2] >= 0 } {
        lappend reflect_huddle_cmds number string boolean
        # huddle package >= 0.2, use builtin strip & dump, bool & number types builtin
        proc ${parent_ns}_impl_huddle_strip {node} {
            return [huddle strip $node]
        }
        proc ${parent_ns}_huddle_wrap_impl {tag spec} {
            return [huddle wrap [list $tag $spec]]
        }
        proc ${parent_ns}_impl_json_dump {huddle_object {offset "  "} {newline "\n"} {begin ""}} {
            return [huddle jsondump $huddle_object $offset $newline $begin]
        }

    } else {
        # huddle package version < 0.2, add number & boolean types, overload strip & dump (use newer version)
        proc ${parent_ns}_impl_huddle_strip {node} {
            if { [llength $node] != 2 } { return "" }
            foreach {head value} $node break
            if { $head eq "HUDDLE" } {
                return [_huddle_strip $value]
            }
            if { $head eq "s" } {
                return $value
            }
            if {[info exists ::huddle::types(type:$head)]} {
                if {$::huddle::types(node:$head) eq "parent"} {
                    return [$::huddle::types(callback:$head) strip $value]
                } else {
                    return $value
                }
            }
            error "_huddle_strip problem: $node"
        }

        proc ${parent_ns}_huddle_wrap_impl {tag spec} {
            return [huddle wrap $tag $spec]
        }

        proc ${parent_ns}_impl_json_dump {huddle_object {offset "  "} {newline "\n"} {begin ""}} {
            # Copied from huddle.tcl v0.1.5
            variable types
            set nextoff "$begin$offset"
            set nlof "$newline$nextoff"
            set sp " "
            if {[string equal $offset ""]} {set sp ""}

            set type [huddle type $huddle_object]
            switch -- $type {
                bool -
                boolean -
                integer -
                num -
                number -
                null {
                    #foreach {head value} $huddle_object break
                    #return $value
                    return [_huddle_strip $huddle_object]
                }

                s -
                string {
                    #foreach {head data} $huddle_object break
                    set data [_huddle_strip $huddle_object]
                    # JSON permits only oneline string
                    set data [string map {
                            \n \\n
                            \t \\t
                            \r \\r
                            \b \\b
                            \f \\f
                            \\ \\\\
                            \" \\\"
                            / \\/
                        } $data
                    ]
                return "\"$data\""
                }

                list {
                    set inner {}
                    set len [huddle llength $huddle_object]
                    if { !$len } { return "\[\]" }
                    for {set i 0} {$i < $len} {incr i} {
                        set subobject [huddle get $huddle_object $i]
                        lappend inner [_impl_json_dump $subobject $offset $newline $nextoff]
                    }
                    if {[llength $inner] == 1} {
                        return "\[[lindex $inner 0]\]"
                    }

                    return "\[$nlof[join $inner ,$nlof]$newline$begin\]"
                }

                dict {
                    set inner {}
                    foreach {key} [huddle keys $huddle_object] {
                        set hobj [huddle get $huddle_object $key]
                        set dump [_impl_json_dump $hobj $offset $newline $nextoff]
                        lappend inner [subst {"$key":$sp$dump}]
                    }
                    if {[llength $inner] == 1} {
                        return $inner
                    }
                    return "\{$nlof[join $inner ,$nlof]$newline$begin\}"
                }

                default {
                    #puts "DBG: jsondump: type=$type data=$data"
                    return [$types(callback:$type) jsondump $data $offset $newline $nextoff]
                }
            }
        }
    }

    foreach cmd $reflect_huddle_cmds {
        proc ${parent_ns}json_$cmd args "huddle $cmd {*}\$args"
    }
}; # end json_init

proc _huddle_strip {node} {
    return [_impl_huddle_strip $node]
}

proc _huddle_is_empty {huddle_object} {
    set type [huddle type $huddle_object]
    switch -- $type {
        s -
        string {
            return [expr [string length [_huddle_strip $huddle_object]] == 0]
        }
        list {
            return [expr [huddle llength $huddle_object] == 0]
        }
        dict {
            return [expr [llength [huddle keys $huddle_object]] == 0]
        }
        default {
            return false
        }
    }
}

proc _huddle_wrap {value spec is_empty_var {debug false}} {
    upvar $is_empty_var is_empty
    set is_empty false

    set check_empty [string match -nocase "*-e" $spec]
    if { ![huddle isHuddle $value] } {
        if { $check_empty } {
            set is_empty [string equal $value ""]
            if { $debug } { puts "DBG: not huddle is_empty=$is_empty spec=$spec value=$value" }
            if { $is_empty } { return "" }
        }
        set spec [string tolower $spec]
        set is_container false
        if { $spec eq "" || [string match "str*" $spec] } {
            return [_huddle_wrap_impl s $value]
        } elseif { [string match "bool*" $spec] } {
            set subcmd "boolean"
        } elseif { [string match "list*" $spec] } {
            set is_container true
            set subcmd "list"
        } elseif { [string match "dict*" $spec] || [string match "create" $spec] } {
            set is_container true
            set subcmd "create"
        } elseif { [string match "num*" $spec] || [string match "int*" $spec] || [string match "float*" $spec] || [string match "double*" $spec] } {
            # ETP: workaround problem parsing JSON numbers
            return [_huddle_wrap_impl s $value]
            set subcmd "number"
        } else {
            error "Unknown huddle spec type: $spec"
        }
        if { $debug } { puts "DBG: not huddle is_empty=$is_empty spec=$spec is_container=$is_container subcmd=$subcmd value=$value" }
        #puts "DBG: huddle_wrap is_container=$is_container spec=$spec value=$value"
        if { $is_container } {
            set newvalue [huddle $subcmd {*}$value]
        } else {
            set newvalue [huddle $subcmd $value]
        }
        if { $debug } { puts "DBG: not huddle is_empty=$is_empty spec=$spec is_container=$is_container subcmd=$subcmd value=$value newvalue=$newvalue" }
        set value $newvalue
    } elseif { $check_empty } {
        if { $debug } { puts "DBG: is huddle spec=$spec value=$value" }
        set is_empty [_huddle_is_empty $value]
        if { $debug } { puts "DBG: is huddle is_empty=$is_empty spec=$spec value=$value" }
    }
    return $value
}

proc json_pretty {json_txt} {
    json_init
    return [_impl_json_dump [_impl_json_read $json_txt]]
}

proc _impl_json_read {json_txt} {
    return [huddle::json2huddle $json_txt]
}

proc json_read {json_txt} {
    json_init
    return [_impl_json_read $json_txt]
}

proc json_dump {huddle_object {offset "  "} {newline "\n"} {begin ""}} {
    json_init
    return [_impl_json_dump $huddle_object $offset $newline $begin]
}

proc json_set {huddle_var key value {spec "string"}} {
    upvar $huddle_var hval
    if { ![info exists hval] } {
        error "variable not set: $huddle_var"
    }
    if { ![huddle isHuddle $hval] } {
        error "not a huddle: $huddle_var=$hval"
    }
    set wrapped [_huddle_wrap $value $spec is_empty]
    if { !$is_empty } {
        huddle set hval $key $wrapped
    } ; #else { puts "DBG: skipping empty $key" }
}

proc json_lappend {huddle_var value {spec "string"}} {
    upvar $huddle_var hval
    if { ![info exists hval] } {
        error "variable not set: $huddle_var"
    }
    if { ![huddle isHuddle $hval] } {
        error "not a huddle: $huddle_var=$hval"
    }
    set wrapped [_huddle_wrap $value $spec is_empty]
    if { !$is_empty } {
        huddle append hval $wrapped
    }
}

proc json_dict {args} {
    return [huddle create {*}$args]
}
proc json_list {args} {
    return [huddle list {*}$args]
}
proc json_size {huddle_val} {
    return [huddle llength $huddle_val]
}
proc json_keys {huddle_val} {
  return [huddle keys $huddle_val]
}
proc json_exists {huddle_val args} {
  if { ![llength $args] } { error "Expected at least 2 arguments to json_exists <json> <key1> \[key2\] ...\[keyN\]" }
  if { [catch {json_get $huddle_val {*}$args}] } { return false }
  return true
}
proc json_get {huddle_val args} {
  if { ![llength $args] } { error "Expected at least 2 arguments to json_get <json> <key1> \[key2\] ...\[keyN\]" }
  return [huddle get $huddle_val {*}$args]
}
proc json_get_default {huddle_val args} {
    if { [llength $args] < 2 } { error "Expected at least 3 arguments to json_get_default <json> <key1> \[key2\] ...\[keyN\] <default-value>" }
    set keys [lrange $args 0 end-1]
    if { [catch {json_get $huddle_val {*}$keys} res] } {
      set res [lindex $args end]; # default
    }
    return $res
}
proc json_lindex {huddle_val index} {
  return [huddle get $huddle_val $index]
}
proc json_type {huddle_val} {
  return [huddle type $huddle_val]
}
proc json_is_container {huddle_val} {
  set type [json_type $huddle_val]
  return [expr {$type eq "list" || $type eq "dict"}]
}

# Convert JSON to native dict/list structure
proc json_to_tcl {hval} {
  if { ![huddle isHuddle $hval] } {
    return $hval
  }
  return [_huddle_strip $hval]
}

proc json_set_int {jobj_var key val} {
    upvar $jobj_var jobj
    # ETP: workaround problem parsing JSON numbers
    return [json_set jobj $key $val]
    if { $val ne "" && [string is integer $val] } {
        json_set jobj $key $val integer
    } else {
        json_set jobj $key $val
    }
}

proc import_commands {ns} {
    ::AP::JSON_UTILS::json_init
    if { ![llength [info procs ${ns}::json_set]] } {
        namespace eval $ns {
            namespace import ::AP::JSON_UTILS::*
        }
    }
}

proc is_huddle_simple_container {huddle_node} {
  set type [huddle type $huddle_node]
  if { $type eq "dict" } {
    foreach value_name [huddle keys $huddle_node] {
      if { [json_is_container [huddle get $huddle_node $value_name]] } {
        return false
      }
    }
  } elseif { $type eq "list" } {
    set len [huddle llength $huddle_node]
    for {set idx 0} {$idx < $len} {incr idx} {
      if { [json_is_container [huddle get $huddle_node $idx]] } {
        return false
      }
    }
  } else {
    return false
  }
  return true
}

proc huddle2xml_add_container {xml_node container_name} {
  if { [string index $container_name end] ne "s" } {
    # add trailing 's'
    append container_name s
  }
  return [AP::ap_dom_document_createElement $xml_node $container_name]
}

proc huddle2xml_add_child {huddle_node xml_node container_name value_name} {
  set type [huddle type $huddle_node]
  if { $container_name ne "" } {
    regsub {s$} $container_name {} container_name; # remove trailing 's'
    set xml_node [AP::ap_dom_document_createElement $xml_node $container_name]
    if { $value_name ne "" } {
      $xml_node setAttribute "${container_name}Name" $value_name
      set value_name ""
    }
  }
  if { $type eq "list" || $type eq "dict" } {
    #set value_name $container_name
    huddle2xml_recursive $huddle_node $xml_node $value_name ""
  } else {
    huddle2xml_add_scalar $huddle_node $xml_node $value_name
  }
}

proc huddle2xml_add_scalar {huddle_node xml_node value_name} {
    if { $value_name eq "" } {
      AP::ap_dom_createValueNode $xml_node [huddle strip $huddle_node]
    } else {
      $xml_node setAttribute $value_name [huddle strip $huddle_node]
    }
}

proc huddle2xml_recursive {huddle_node xml_node {container_name ""} {value_name ""} {create_top_container true}} {
  set type [huddle type $huddle_node]
  if { $type eq "null" } {
    return
  }

  if { $create_top_container && $container_name ne "" && ($type eq "list" || $type eq "dict") } {
    set xml_node [huddle2xml_add_container $xml_node $container_name]
  }

  if { $type eq "dict" } {
    foreach value_name [huddle keys $huddle_node] {
      set huddle_child [huddle get $huddle_node $value_name]
      huddle2xml_add_child $huddle_child $xml_node $container_name $value_name
    }
  } elseif { $type eq "list" } {
    set len [huddle llength $huddle_node]
    for {set idx 0} {$idx < $len} {incr idx} {
      set huddle_child [huddle get $huddle_node $idx]
      huddle2xml_add_child $huddle_child $xml_node $container_name ""
    }
  } else {
    huddle2xml_add_scalar $huddle_node $xml_node $value_name
  }
}

proc huddle2xml {huddle_node xml_node {container_name ""} {create_top_container true}} {
  return [huddle2xml_recursive $huddle_node $xml_node $container_name "" $create_top_container]
}

namespace export json_*

} ; # end ::AP::JSON_UTILS namespace

proc ::dict_get_default {adict args} {
    if { [llength $args] < 2 } {
        error "Expected at least 3 arguments to dict_get_default <dict> <dict-key1> \[dict-key2\] ...\[dict-keyN\] <default-value>: got [expr [llength $args]+1] arguments: [list $adict] $args"
    }
    set keys [lrange $args 0 end-1]
    if { [dict exists $adict {*}$keys] } {
        return [dict get $adict {*}$keys]
    }
    return [lindex $args end]; # return default
}

proc ::AP::jsonfile2dict {json_file} {
    set fh [open $json_file r]
    set rawjson [read $fh]
    close $fh
    return [AP::json2dict $rawjson]
}

proc ::AP::json2dict {json_text} {
    if { ![string length [string trim $json_text]] } {
        return [dict create]
    }
    package require json
    return [json::json2dict $json_text]
}
